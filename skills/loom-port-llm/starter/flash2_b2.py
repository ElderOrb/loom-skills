"""Key-split flash loop with two query heads per wave for the gfx12 head_dim 512 twin (gen_flash2s.py, FLASH2_B=2; llm_config rdna4_f16_ffn sets it).

The staged twin lets the four waves of a query head compute every 16-key S tile together (each over its 128 dims, an f32 partial exchange,
four redundant softmaxes) and every wave streams its own K / V^T fragments for ONE head. This loop reorganises the work the way llama.cpp's
Vulkan flash_attn_cm1 does and shares the K / V fragments between two heads:
  * a block is 64 keys; wave j of a workgroup computes the S^T tile of keys 16 j .. 16 j + 15 over ALL 512 dims (32 WMMAs per head; the K
    fragments straight from the paged KV arena as four 128-byte loads per lane, the Q^T fragments from workgroup memory where the four
    waves staged them once) for BOTH heads of the workgroup's head pair, so a K fragment feeds two WMMAs;
  * ONE softmax per S tile: the column maxima of the four tiles of a head are exchanged (one barrier), every wave derives the same running
    max, exponentiates its own tile and publishes P^T (f16, the rhs fragment as it is) and its row sum;
  * PV: every wave reads the four P^T tiles of both heads (second barrier) and accumulates O^T of ITS 128 dims over all 64 keys; a V^T
    fragment (global_load_tr_b128 from the arena) feeds both heads' WMMAs.
Per 64 keys and head pair: two barriers, 128 WMMAs per wave. A workgroup is the four waves of a head pair (128 lanes); the exchange buffers
are single (a wave cannot reach the next block's store before every wave has finished reading this block's), so the workgroup needs 40 KB of
workgroup memory and three of them share a WGP. Keys past the tile's last written page re-read the last page (finite, fully masked, weight 0);
the partial kernel zeroes V^T for positions past its last written key. The first head's output epilogue is repeated for the second head.
Measured on E2B (R9700, vs the staged twin): prefill +7 / +12 / +19 % at 8k / 16k / 32k.
"""
import re


def rewrite_b2(text, kvh, d, partial, qh=None, nh=2, opts=None):
    """d = 512 (gemma4 global layers: 8 query heads per KV head) or 256 (Qwen3.5-class 27B: 6 per KV head, three head pairs per KV head).
    A wave owns d / 4 output dims (HALF of gen_flash2s.py: 128 / 64) and sums its 16 keys' S^T over all d dims."""
    assert d in (256, 512) and nh in (2, 3)
    waves = 4
    SUF = ["", "B", "C"][:nh]                                   # per head: names of its running max / sum / S tile / epilogue ("" = the first head)
    AP = ["a", "b", "cc"][:nh]                                  # per head: its accumulator names
    OP = ["o", "oB", "oC"][:nh]                                 # per head: its loop results
    HV = ["%head", "%head1", "%head2"][:nh]
    qh = qh or kvh * 8
    R = kvh * d
    NF = d // 16
    NO = d // 64                   # output fragments of a wave per head (8 / 4)
    NKW = NF // 8                  # 64-dim K loads per lane for one 16-key S tile (4 / 2)
    FPW = NF // waves              # Q^T fragments each wave stages (8 / 4)
    QP = NF * 8 + 8
    pos_bytes = int(re.search(r"%pos_bytes = index\.constant (\d+) : offset", text).group(1))
    marker = "  // Partial tile: O^T / l" if partial else "  // O^T / l, gate, store through"
    text = re.sub(r"^ *%k_bytes = .*\n *%k_tile = .*\n *%k_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%vt_bytes = .*\n *%vt_tile = .*\n *%vt_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%s_bytes = .*\n *%s_tile = .*\n *%s_view = .*\n", "", text, flags=re.M)
    h0 = text.index("  %m_final, %l_final,")
    h1 = text.index(marker)
    epi = text[h1:]
    body_end = epi.index("  }\n  kernel.return")        # the epilogue ends at the scf.if's closing brace
    epilogue, tail = epi[:body_end], epi[body_end:]
    # --- the other heads' epilogues: every name the epilogue defines gets the head's suffix; the loop results and the head index are renamed
    defined = set(re.findall(r"^\s*(%\w+)(?:, %\w+)* =", epilogue, flags=re.M))
    defined |= set(m for line in re.findall(r"^\s*((?:%\w+, )+%\w+) =", epilogue, flags=re.M) for m in line.split(", "))
    epis = []
    for hh in range(1, nh):
        epi_b = epilogue
        for nm in sorted(defined, key=len, reverse=True):
            epi_b = re.sub(re.escape(nm) + r"\b(?!\w)", nm + SUF[hh], epi_b)
        for k in range(NO):
            epi_b = re.sub(rf"%o{k}\b(?!\w)", f"%{OP[hh]}{k}", epi_b)
        epi_b = re.sub(r"%l_final\b(?!\w)", f"%l{SUF[hh]}_final", epi_b)
        epi_b = re.sub(r"(?<![\w%])%head\b(?!\w)", HV[hh], epi_b)
        epi_b = re.sub(r"%head\b(?!\w)", HV[hh], epi_b)
        epis.append(epi_b)
    # Q^T staging of both heads
    cfpw = "%fb_c8" if FPW == 8 else "%fb_cfpw"   # d 512 keeps its original text (the gemma4 program stays byte-identical)
    qst = ("  %fb_c8 = index.constant 8 : index\n  %fb_c64 = index.constant " + str(FPW * 8) + " : index\n  %fb_c256 = index.constant " + str(d // 2) + " : index\n"
           + ("" if FPW == 8 else "  %fb_cfpw = index.constant " + str(FPW) + " : index\n") +
           "  %fb_dh8 = index.mul %dim_half, " + cfpw + " : index\n  %fb_dh64 = index.mul %dim_half, %fb_c64 : index\n"
           "  %fb_h256 = index.mul %lane_half, %fb_c256 : index\n  %fb_qb = index.add %fb_h256, %fb_dh64 : index\n"
           + "".join(f"  %head{hh}_v = index.add %head, %c{hh} : index\n  %head{hh} = index.assume %head{hh}_v [range(%head{hh}_v, {hh}, {qh - 1})] : index\n" for hh in range(1, nh)))
    for hh, hn in enumerate(HV):
        for c in range(FPW):
            qst += (f"  %fqs{hh}_{c}_v = index.add %fb_dh8, %e{c} : index\n  %fqs{hh}_{c} = index.assume %fqs{hh}_{c}_v [range(%fqs{hh}_{c}_v, 0, {NF - 1})] : index\n"
                    f"  %fqe{hh}_{c}_v = index.mul %fqs{hh}_{c}, %fb_c8 : index\n  %fqe{hh}_{c} = index.assume %fqe{hh}_{c}_v [range(%fqe{hh}_{c}_v, 0, {QP - 8})] : index\n"
                    f"  %fqn{hh}_{c}_o = index.constant {8 * c} : index\n  %fqn{hh}_{c}_v = index.add %fb_qb, %fqn{hh}_{c}_o : index\n"
                    f"  %fqn{hh}_{c} = index.assume %fqn{hh}_{c}_v [range(%fqn{hh}_{c}_v, 0, {d - 8})] : index\n"
                    f"  %fqraw{hh}_{c} = vector.load %q_view[%q_token, {hn}, %fqn{hh}_{c}] : view<[%tokens]x{qh}x{d}xf32> -> vector<8xf32>\n"
                    f"  %fqh{hh}_{c} = vector.fptrunc %fqraw{hh}_{c} : vector<8xf32> to vector<8xf16>\n"
                    f"  vector.store %fqh{hh}_{c}, %ql_view[%c{hh}_i, %lane, %fqe{hh}_{c}] : vector<8xf16>, view<{nh}x32x{QP}xf16>\n")
    qst += "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    alloc = (f"  %c0_i = index.constant 0 : index\n  %c1_i = index.constant 1 : index\n" + ("  %c2_i = index.constant 2 : index\n" if nh == 3 else "") +
             f"  %ql_bytes = index.constant {nh * 32 * QP * 2} : offset\n  %ql_tile = buffer.alloca<workgroup> align(16) %ql_bytes : buffer\n"
             f"  %ql_view = buffer.view %ql_tile[%origin] : buffer -> view<{nh}x32x{QP}xf16>\n"
             f"  %pl_bytes = index.constant {nh * waves * 32 * 16} : offset\n  %pl_tile = buffer.alloca<workgroup> align(16) %pl_bytes : buffer\n"
             f"  %pl_view = buffer.view %pl_tile[%origin] : buffer -> view<1x{nh}x{waves}x32x8xf16>\n"
             f"  %mx_bytes = index.constant {nh * waves * 32 * 4} : offset\n  %mx_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %mx_view = buffer.view %mx_tile[%origin] : buffer -> view<1x{nh}x{waves}x32xf32>\n"
             f"  %ls_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %ls_view = buffer.view %ls_tile[%origin] : buffer -> view<1x{nh}x{waves}x32xf32>\n")
    defs = f"""  %vt_g_c1 = index.constant 1 : index
  %vt_g_crow = index.constant {R} : index
  %vt_g_layout = encoding.layout.strided [%vt_g_c1, %vt_g_crow] : encoding<layout>
  %k_g_layout = encoding.layout.strided [%vt_g_crow, %vt_g_c1] : encoding<layout>
  %vd_cd = index.constant {d} : index
  %vd_kvd = index.mul %kv, %vd_cd : index
  %vd_dk = index.add %vd_kvd, %dim_base : index
  %vd_c2b = index.constant 2 : offset
  %kv_boff = index.scale %vd_kvd, %vd_c2b : index, offset -> offset
  %vd_boff = index.scale %vd_dk, %vd_c2b : index, offset -> offset
  %kh_lane0 = scalar.constant 0 : i32
  %kh_rec_stride = index.constant 40 : offset
  %kh_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %kh_rec_off = index.scale %kh_row, %kh_rec_stride : index, offset -> offset
  %kh_records = buffer.view %origins[%kh_rec_off] : buffer -> view<5xi64>
  %kh_map64 = view.load %kh_records[4] : view<5xi64> -> i64
  %kh_map_b = scalar.assume %kh_map64 [range(%kh_map64, 0, 9223372036854775807)] : i64
  %kh_map_origin = index.cast %kh_map_b : i64 to offset
  %kh_map = buffer.view %origins[%kh_map_origin] : buffer -> view<[%blocks]xi32>
  %kh_last = index.sub %blocks, %c1 : index
  %kh_pp = index.mul %k_plane, %pool : index
  %kh_ppo = index.scale %kh_pp, %pos_bytes : index, offset -> offset
  %kb_o = index.add %kh_ppo, %kv_boff : offset
  %vb_o0 = index.add %kh_ppo, %vd_boff : offset
  %vb_o = index.add %vb_o0, %v_plane_delta : offset
  %pl_rows_m = index.sub %pool, %c16 : index
  %k_pv = buffer.view %arena[%kb_o] : buffer -> view<[%pool]x{d}xf16, %k_g_layout>
  %pg_live_v = index.div %pos0, %c16 : index
  %pg_live = index.assume %pg_live_v [range(%pg_live_v, 0, %kh_last), lt(%pg_live_v, %blocks)] : index
  %c64 = index.constant 64 : index
  %c63 = index.constant 63 : index
  %zv8 = vector.constant 0.0 : vector<8xf16>
  %nblk_round = index.add %key_count, %c63 : index
  %nblk = index.div %nblk_round, %c64 : index
  %dh16 = index.mul %dim_half, %c16 : index
  %dh16_i32 = index.cast %dh16 : index to i32
  %fb_is0 = index.cmp eq, %dim_half, %c0 : index
  %fb_is1 = index.cmp eq, %dim_half, %c1 : index
  %fb_is2 = index.cmp eq, %dim_half, %c2 : index
  %fb_wv = index.assume %wave [range(%wave, 0, 3)] : index
"""
    for f in range(NF):
        defs += f"  %fq{f} = index.constant {f} : index\n  %kcol{f} = index.constant {16 * f} : index\n"
    if partial:
        pass
    ABC = "ABC"[:nh]
    acc = ", ".join(f"%{AP[hh]}{k} = %init : vector<8xf32>" for hh in range(nh) for k in range(NO))
    outs = ", ".join(f"%{OP[hh]}{k}" for hh in range(nh) for k in range(NO))
    ty16 = ", ".join(["vector<8xf32>"] * (nh * NO))
    mls = ", ".join(f"%m{SUF[hh]}_final, %l{SUF[hh]}_final" for hh in range(nh))
    mli = ", ".join(f"%m{SUF[hh]} = %neg_big : f32, %l{SUF[hh]} = %zero_f : f32" for hh in range(nh))
    mlt = ", ".join(["f32, f32"] * nh)
    L = []
    A = L.append
    A(f"  {mls}, {outs} = scf.for %bt = [%c0 to %nblk step %c1]({mli}, {acc}) -> ({mlt}, {ty16}) {{")
    A("    %bt64 = index.mul %bt, %c64 : index\n    %blk_first = index.add %bt64, %key_low : index\n    %blk_pg0 = index.div %blk_first, %c16 : index\n"
      "    %par_i = index.rem %bt, %c2 : index\n    %par = index.assume %par_i [range(%par_i, 0, 1)] : index")
    for j in range(4):
        A(f"""    %pg{j}_a = index.add %blk_pg0, %fq{j} : index
    %pg{j}_c = index.min %pg{j}_a, %pg_live : index
    %pg{j} = index.assume %pg{j}_c [range(%pg{j}_c, 0, %kh_last), lt(%pg{j}_c, %blocks)] : index
    %ph{j}_r = view.load %kh_map[%pg{j}] : view<[%blocks]xi32> -> i32
    %ph{j}_s = scalar.addi %ph{j}_r, %kh_lane0 : i32
    %ph{j}_b = scalar.assume %ph{j}_s [range(%ph{j}_s, 0, 65535)] : i32
    %ph{j}_i = index.cast %ph{j}_b : i32 to index
    %rw{j}_v = index.mul %ph{j}_i, %c16 : index
    %rw{j} = index.assume %rw{j}_v [range(%rw{j}_v, 0, 1048560), lt(%rw{j}_v, %pl_rows_m)] : index""")
    A("    %rk_a = scf.select %fb_is2, %rw2, %rw3 : index\n    %rk_b = scf.select %fb_is1, %rw1, %rk_a : index\n    %rk_c = scf.select %fb_is0, %rw0, %rk_b : index\n    %rk = index.assume %rk_c [range(%rk_c, 0, 1048560), lt(%rk_c, %pl_rows_m)] : index")
    opts = opts or {}
    VPF = opts.get("vpf", "")      # "" = the V^T loads inside the PV loop; "top" / "mid" / "top2": issued at the block start / after the QK loop / half each

    def vloads(jjs):
        out = ""
        for jj in jjs:
            out += (f"    %vj{jj}_ro = index.scale %rw{jj}, %pos_bytes : index, offset -> offset\n    %vj{jj}_o = index.add %vb_o, %vj{jj}_ro : offset\n"
                    f"    %vj{jj} = buffer.view %arena[%vj{jj}_o] : buffer -> view<{d // 4}x16xf16, %vt_g_layout>\n")
            for k in range(NO):
                out += f"    %vc{jj}_{k} = vector.fragment.load<lhs> %vj{jj}[%kcol{k}, %c0] shape [%c16, %c16] : view<{d // 4}x16xf16, %vt_g_layout> -> vector<8xf16>\n"
        return out
    if VPF in ("top", "top2"):
        A(vloads(range(4) if VPF == "top" else (0, 1)))
    A("    %kw_lr_v = index.add %rk, %lane_col : index\n    %kw_lr = index.assume %kw_lr_v [range(%kw_lr_v, 0, 1048575), lt(%kw_lr_v, %pool)] : index\n"
      f"    %kw_c256 = index.constant {d // 2} : index\n    %kw_h = index.mul %lane_half, %kw_c256 : index\n    %c64k = index.constant 64 : index\n    %fq8c = index.constant 8 : index")
    sinit = ", ".join(f"%s{x} = %init : vector<8xf32>" for x in ABC)
    sres = ", ".join(f"%s{x}_f" for x in ABC)
    sty = ", ".join(["vector<8xf32>"] * nh)
    A(f"    {sres} = scf.for %kw = [%c0 to %c{NKW} step %c1]({sinit}) -> ({sty}){{")
    A(f"""      %kw_a = index.assume %kw [range(%kw, 0, {NKW - 1})] : index
      %kw_o = index.mul %kw_a, %c64k : index
      %kw_c = index.add %kw_h, %kw_o : index
      %kw_ci = index.assume %kw_c [range(%kw_c, 0, {d - 64})] : index
      %kwv = vector.load %k_pv[%kw_lr, %kw_ci] : view<[%pool]x{d}xf16, %k_g_layout> -> vector<64xf16>
      %kw_f0 = index.mul %kw_a, %fq8c : index
      %kqo_v = index.mul %kw_f0, %fq8c : index
      %kqo = index.assume %kqo_v [range(%kqo_v, 0, {QP - 64})] : index""" + "".join(f"""
      %qrw{x} = vector.load %ql_view[%c{hh}_i, %lane, %kqo] : view<{nh}x32x{QP}xf16> -> vector<64xf16>""" for hh, x in enumerate(ABC)))
    cur = [f"%s{x}" for x in ABC]
    for sl in range(8):
        A(f"""      %kfs{sl} = vector.slice %kwv[{8 * sl}] : vector<64xf16> -> vector<8xf16>
      %kf{sl} = vector.fragment<lhs> %kfs{sl} shape [%c16, %c16] : vector<8xf16>
""" + "".join(f"      %qr{x}{sl} = vector.slice %qrw{x}[{8 * sl}] : vector<64xf16> -> vector<8xf16>\n" for x in ABC)
          + "".join(f"      %qb{x}{sl} = vector.fragment<rhs> %qr{x}{sl} shape [%c16, %c16] : vector<8xf16>\n" for x in ABC)
          + "\n".join(f"      %sq{x}{sl} = vector.mma %kf{sl}, %qb{x}{sl}, {cur[hh]} : vector<8xf16>, vector<8xf16>, vector<8xf32>" for hh, x in enumerate(ABC)))
        cur = [f"%sq{x}{sl}" for x in ABC]
    A(f"      scf.yield {', '.join(cur)} : {sty}\n    }}")
    if VPF in ("mid", "top2"):
        A(vloads(range(4) if VPF == "mid" else (2, 3)))
    A(f"""    %tf32 = index.cast %blk_first : index to i32
    %kbase32 = scalar.addi %tf32, %dh16_i32 : i32
    %base_v = vector.splat %kbase32 : vector<8xi32>
    %kpos = vector.addi %base_v, %row_off : vector<8xi32>
    %causal = vector.cmpi sle, %kpos, %qp_v : vector<8xi32> -> vector<8xi1>
    %inwin = vector.cmpi sge, %kpos, %low_v : vector<8xi32> -> vector<8xi1>
    %winok = vector.ori %inwin, %ns_v : vector<8xi1>
    %valid = vector.andi %causal, %winok : vector<8xi1>""")
    for hh, s_ in enumerate(SUF):
        A(f"""    %sc{s_} = vector.mulf %s{ABC[hh]}_f, %scale_v : vector<8xf32>
    %masked{s_} = vector.select %valid, %sc{s_}, %neg_v : vector<8xf32>
    %lmax{s_} = vector.reduce<maxnumf> %masked{s_}, %neg_big : vector<8xf32>, f32
    %omax{s_}, %omax{s_}_u = kernel.subgroup.shuffle<xor> %lmax{s_}, %exchange, %width : f32, i32, i32
    %tmax{s_} = scalar.maxnumf %lmax{s_}, %omax{s_} : f32
    view.store %tmax{s_}, %mx_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x{nh}x{waves}x32xf32>""")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    for hh, s_ in enumerate(SUF):
        for i in range(4):
            A(f"    %mx{s_}{i} = view.load %mx_view[%c0_i, %c{hh}_i, %fq{i}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32")
        A(f"""    %mxa{s_} = scalar.maxnumf %mx{s_}0, %mx{s_}1 : f32
    %mxb{s_} = scalar.maxnumf %mx{s_}2, %mx{s_}3 : f32
    %bm{s_} = scalar.maxnumf %mxa{s_}, %mxb{s_} : f32
    %m{s_}_new = scalar.maxnumf %m{s_}, %bm{s_} : f32
    %m{s_}_shift = scalar.subf %m{s_}, %m{s_}_new : f32
    %alpha{s_} = scalar.expf<afn> %m{s_}_shift : f32
    %m{s_}_new_v = vector.splat %m{s_}_new : vector<8xf32>
    %sh{s_} = vector.subf %masked{s_}, %m{s_}_new_v : vector<8xf32>
    %pe{s_} = vector.expf<afn> %sh{s_} : vector<8xf32>
    %p{s_} = vector.select %valid, %pe{s_}, %zero8 : vector<8xf32>
    %ls{s_} = vector.reduce<addf> %p{s_}, %zero_f : vector<8xf32>, f32
    %osum{s_}, %osum{s_}_u = kernel.subgroup.shuffle<xor> %ls{s_}, %exchange, %width : f32, i32, i32
    %tsum{s_} = scalar.addf %ls{s_}, %osum{s_} : f32
    %pt_h{s_} = vector.fptrunc %p{s_} : vector<8xf32> to vector<8xf16>
    vector.store %pt_h{s_}, %pl_view[%c0_i, %c{hh}_i, %fb_wv, %lane, %c0] : vector<8xf16>, view<1x{nh}x{waves}x32x8xf16>
    view.store %tsum{s_}, %ls_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x{nh}x{waves}x32xf32>""")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    for hh, s_ in enumerate(SUF):
        for j in range(4):
            A(f"    %lv{s_}{j} = view.load %ls_view[%c0_i, %c{hh}_i, %fq{j}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32")
        A(f"""    %lt01{s_} = scalar.addf %lv{s_}0, %lv{s_}1 : f32
    %lt012{s_} = scalar.addf %lt01{s_}, %lv{s_}2 : f32
    %lt{s_} = scalar.addf %lt012{s_}, %lv{s_}3 : f32
    %l_scaled{s_} = scalar.mulf %l{s_}, %alpha{s_} : f32
    %l{s_}_new = scalar.addf %l_scaled{s_}, %lt{s_} : f32
    %alpha{s_}_v = vector.splat %alpha{s_} : vector<8xf32>""")
    if partial:
        A("    %pos_hi_v = vector.splat %pos_hi_i32 : vector<8xi32>")
    PA = ["pa", "pb", "pcc"]
    BN = ["bn", "bnB", "bnC"]
    ON = ["on", "onB", "onC"]
    for k in range(NO):
        A("\n".join(f"    %os{SUF[hh]}{k} = vector.mulf %{AP[hh]}{k}, %alpha{SUF[hh]}_v : vector<8xf32>" for hh in range(nh)))
    if VPF:   # straight-line PV over the four pages of the block (V^T fragments already loaded)
        cur = {(hh, k): f"%os{SUF[hh]}{k}" for hh in range(nh) for k in range(NO)}
        for jj in range(4):
            for hh, x in enumerate(ABC):
                A(f"    %pr{x}{jj} = vector.load %pl_view[%c0_i, %c{hh}_i, %fq{jj}, %lane, %c0] : view<1x{nh}x{waves}x32x8xf16> -> vector<8xf16>\n"
                  f"    %pf{x}{jj} = vector.fragment<rhs> %pr{x}{jj} shape [%c16, %c16] : vector<8xf16>")
            if partial:
                A(f"""    %vkc{jj} = scalar.constant {16 * jj} : i32
    %vkb{jj} = scalar.addi %tf32, %vkc{jj} : i32
    %vkb{jj}_v = vector.splat %vkb{jj} : vector<8xi32>
    %vkp{jj} = vector.addi %vkb{jj}_v, %row_off : vector<8xi32>
    %vok{jj} = vector.cmpi sle, %vkp{jj}, %pos_hi_v : vector<8xi32> -> vector<8xi1>""")
            for k in range(NO):
                vname = f"%vc{jj}_{k}"
                if partial:
                    A(f"    %vs{jj}_{k} = vector.select %vok{jj}, %vc{jj}_{k}, %zv8 : vector<8xf16>\n    %vm{jj}_{k} = vector.fragment<lhs> %vs{jj}_{k} shape [%c16, %c16] : vector<8xf16>")
                    vname = f"%vm{jj}_{k}"
                for hh, x in enumerate(ABC):
                    nm = f"%{ON[hh]}{k}" if jj == 3 else f"%{BN[hh]}{jj}_{k}"
                    A(f"    {nm} = vector.mma {vname}, %pf{x}{jj}, {cur[(hh, k)]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                    cur[(hh, k)] = nm
    else:
        pvin = ", ".join(f"%{PA[hh]}{k} = %os{SUF[hh]}{k} : vector<8xf32>" for hh in range(nh) for k in range(NO))
        pvout = ", ".join(f"%{ON[hh]}{k}" for hh in range(nh) for k in range(NO))
        A(f"    {pvout} = scf.for %jj = [%c0 to %c4 step %c1]({pvin}) -> ({ty16}) unroll {{")
        VJ = """      %rj_a = scf.select %jj_c2, %rw2, %rw3 : index
      %rj_b = scf.select %jj_c1, %rw1, %rj_a : index
      %rj_c = scf.select %jj_c0, %rw0, %rj_b : index
      %rj = index.assume %rj_c [range(%rj_c, 0, 1048560), lt(%rj_c, %pl_rows_m)] : index
      %vj_ro = index.scale %rj, %pos_bytes : index, offset -> offset
      %vj_o = index.add %vb_o, %vj_ro : offset
"""
        A(f"""      %jj_a = index.assume %jj [range(%jj, 0, 3)] : index
      %jj_c1 = index.cmp eq, %jj, %c1 : index
      %jj_c2 = index.cmp eq, %jj, %c2 : index
      %jj_c0 = index.cmp eq, %jj, %c0 : index
{VJ}      %vj = buffer.view %arena[%vj_o] : buffer -> view<{d // 4}x16xf16, %vt_g_layout>""" + "".join(f"""
      %pr{x} = vector.load %pl_view[%c0_i, %c{hh}_i, %jj_a, %lane, %c0] : view<1x{nh}x{waves}x32x8xf16> -> vector<8xf16>
      %pf{x} = vector.fragment<rhs> %pr{x} shape [%c16, %c16] : vector<8xf16>""" for hh, x in enumerate(ABC)))
        if partial:
            A("""      %jj16 = index.mul %jj_a, %c16 : index
      %jj16_i = index.cast %jj16 : index to i32
      %vkb = scalar.addi %tf32, %jj16_i : i32
      %vkb_v = vector.splat %vkb : vector<8xi32>
      %vkp = vector.addi %vkb_v, %row_off : vector<8xi32>
      %vok = vector.cmpi sle, %vkp, %pos_hi_v : vector<8xi32> -> vector<8xi1>""")
        for k in range(NO):
            A(f"      %vc{k} = vector.fragment.load<lhs> %vj[%kcol{k}, %c0] shape [%c16, %c16] : view<{d // 4}x16xf16, %vt_g_layout> -> vector<8xf16>")
        for k in range(NO):
            vname = f"%vc{k}"
            if partial:
                A(f"      %vs{k} = vector.select %vok, %vc{k}, %zv8 : vector<8xf16>\n      %vm{k} = vector.fragment<lhs> %vs{k} shape [%c16, %c16] : vector<8xf16>")
                vname = f"%vm{k}"
            A("\n".join(f"      %{BN[hh]}{k} = vector.mma {vname}, %pf{x}, %{PA[hh]}{k} : vector<8xf16>, vector<8xf16>, vector<8xf32>" for hh, x in enumerate(ABC)))
        A("      scf.yield " + ", ".join(f"%{BN[hh]}{k}" for hh in range(nh) for k in range(NO)) + f" : {ty16}")
        A("    }")
    A("    scf.yield " + ", ".join(f"%m{SUF[hh]}_new, %l{SUF[hh]}_new" for hh in range(nh)) + ", " + ", ".join(f"%{ON[hh]}{k}" for hh in range(nh) for k in range(NO)) + f" : {mlt}, {ty16}\n  }}\n")
    loop = "".join(x if x.endswith("\n") else x + "\n" for x in L)
    loop = _ablate(loop, (opts or {}).get("abl", ""))
    text = text[:h0] + alloc + defs + qst + loop + epilogue + "".join(epis) + tail
    return text


def _ablate(loop, abl):
    """Timing ablations of the loop (kbench only; results are wrong): kv0 every K / V^T page is page 0 (cache-resident), noqk no QK
    WMMAs (nor K loads), nopv no PV WMMAs (nor V^T loads), nobar no barriers, noexp no exponentials."""
    for a in [x for x in abl.split(",") if x]:
        if a == "kv0":
            loop = re.sub(r"(%rw\d) = index\.assume %rw\d_v \[range\(%rw\d_v,", r"\1 = index.assume %c0 [range(%c0,", loop)
            loop = re.sub(r"lt\(%rw\d_v, %pl_rows_m\)", "lt(%c0, %pl_rows_m)", loop)
        elif a in ("noqk", "nopv"):
            pat = r"^ *(%sq\w+) = vector\.mma \S+, \S+, (%\w+) :.*\n" if a == "noqk" else r"^ *(%bn\w+) = vector\.mma \S+, \S+, (%\w+) :.*\n"
            for m in list(re.finditer(pat, loop, flags=re.M)):
                loop = loop.replace(m.group(0), "")
                loop = re.sub(re.escape(m.group(1)) + r"\b(?!\w)", m.group(2), loop)
        elif a == "nobar":
            loop = re.sub(r"^ *kernel\.barrier.*\n", "", loop, flags=re.M)
        elif a == "noexp":
            loop = re.sub(r"vector\.expf<afn> (%\w+) : vector<8xf32>", r"vector.addf \1, %zero8 : vector<8xf32>", loop)
        else:
            raise ValueError("flash2_b2: unknown ablation " + a)
    return loop
