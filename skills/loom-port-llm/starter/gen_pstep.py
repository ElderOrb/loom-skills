"""Persistent / prologue-fused variants of the int8-activation decode matvec (gen_llm._mv, iq=True) for the drafted verify step.

Opt-in (choice decode.mv.persist, env MV_PERSIST; the AMD MXFP4 node on rdna4); the GGUF fleet never sees these kernels.
Every kernel here is a text transform of the kernel gen_llm._mv generates (the same weight decode, integer dots, butterfly
and partial fold, so each output row is summed exactly as the source kernel sums it), with three independent changes:

  persistent row loop (G)   the grid is min(G, row groups) workgroups and each workgroup loops over row groups
                            vg = id, id + G, ...: the activation is loaded (and, with a prologue, produced) once per
                            workgroup instead of once per row group, and the dispatch has G workgroups instead of thousands.
  several matrices (mats)   row groups [0, rg1) are matrix 1, [rg1, rg1 + rg2) matrix 2 (same input, e.g. gate + up,
                            qkv + z): one launch for projections that read one activation.
  activation prologue       the int8 codes (g_quant_q8's layout and arithmetic) are produced in the lane that consumes
                            them, from:  f32  an f32 row (replaces g_quant_q8),
                                         silu silu(f) * u of two f32 rows (replaces g_silu_mul + g_quant_q8),
                                         norm rmsnorm(x) * w of the residual row (replaces g_post_add_norm's h + g_quant_q8;
                                              the residual add moves into the producing matvec's accumulate store).
                            The prologue ops are g_quant_q8's / g_silu_mul's / g_post_add_norm's ops in the same order, so
                            the codes are bit-identical (kernel tests compare against the launch chain with equality).

Names: g_pmv_<fmt>_k<K>_t<TG>_<prologue>_r<R>[_m2][_p<G>]; launch args (n, [n2,] tokens, accumulate, x, [u | nw], w, ws,
[w2, ws2,] y [, y2]).
"""
import re

PROLOGUES = ("q8", "f32", "silu", "norm")


def _one(text, anchor, new, what):
    assert text.count(anchor) == 1, f"gen_pstep {what}: the matvec template moved ({anchor.strip()[:60]!r} x{text.count(anchor)})"
    return text.replace(anchor, new, 1)


ROW_LOOP = "  scf.for %r = [%c0 to %rows_per step %c1] {\n"
TAIL = "  kernel.return\n}\n"
ROW_BASE = "  %row_base = index.mul %group_id, %rows_per : index\n"
TOKEN_BLOCK = re.compile(
    r"  %q(\d+), %da\1, %xs\1 = scf\.if %use\1 -> \(vector<32xi8>, f32, f32\) \{\n.*?\n  \} else \{\n"
    r"    scf\.yield %zero32i8, %zero_f, %zero_f : vector<32xi8>, f32, f32\n  \}", re.S)


def quant_lines(j, v, halves=4):
    """g_quant_q8's arithmetic on this lane's 32 values %{v}: codes %q{j}, F16-rounded scale %da{j}, code sum %xs{j}. The
    elementwise part runs on `halves` slices (the same per-element ops, so the same codes) to bound the registers the f32
    division needs; amax (maxnum) and the integer code sum are order-free."""
    n = 32 // halves
    t = [f"""  %qa{j} = vector.absf %{v} : vector<32xf32>
  %qamax{j} = vector.reduce<maxnumf> %qa{j}, %zero_f : vector<32xf32>, f32
  %qd{j} = scalar.divf %qamax{j}, %pq_c127 : f32
  %qnz{j} = scalar.cmpf one, %qamax{j}, %zero_f : f32
  %qdv{j} = vector.splat %qd{j} : vector<{n}xf32>
  %qnzv{j} = vector.splat %qnz{j} : vector<{n}xi1>
  %qz{j} = vector.constant 0.0 : vector<{n}xf32>"""]
    parts = []
    for k in range(halves):
        t.append(f"""  %qv{j}_{k} = vector.slice %{v}[{k * n}] : vector<32xf32> -> vector<{n}xf32>
  %qsc{j}_{k} = vector.divf %qv{j}_{k}, %qdv{j} : vector<{n}xf32>
  %qrd{j}_{k} = vector.roundf %qsc{j}_{k} : vector<{n}xf32>
  %qrs{j}_{k} = vector.select %qnzv{j}, %qrd{j}_{k}, %qz{j} : vector<{n}xf32>
  %qi{j}_{k} = vector.fptosi %qrs{j}_{k} : vector<{n}xf32> to vector<{n}xi8>
  scf.schedule.fence""")
        parts.append(f"%qi{j}_{k}")
    while len(parts) > 1:
        w = 32 // len(parts)
        nxt = []
        for i in range(0, len(parts), 2):
            c = f"%qc{j}_{len(parts)}_{i}"
            t.append(f"  {c} = vector.concat<0> {parts[i]}, {parts[i + 1]} : vector<{w}xi8>, vector<{w}xi8> -> vector<{2 * w}xi8>")
            nxt.append(c)
        parts = nxt
    t.append(f"""  %q{j} = vector.bitcast {parts[0]} : vector<32xi8> to vector<32xi8>
  %qqs{j} = vector.dot4i<s8s8> %q{j}, %pq_ones8, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
  %qqsi{j} = vector.reduce<addi> %qqs{j}, %zero_i : vector<8xi32>, i32
  %xs{j} = scalar.sitofp %qqsi{j} : i32 to f32
  %qdh{j} = scalar.fptrunc %qd{j} : f32 to f16
  %da{j} = scalar.extf %qdh{j} : f16 to f32
  scf.schedule.fence""")
    return "\n".join(t)


def _load(j, view, groups, name):
    return f"""  %{name}{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk{j} = index.assume %tok{j}_raw [lt(%tok{j}_raw, %tokens)] : index
    %v = vector.load %{view}[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}"""


def prologue(text, mode, groups, eps=None, count=None):
    """Replace the int8 activation loads of an iq matvec by the producer computed in the lane (module doc)."""
    if mode == "q8":
        return text
    assert mode in PROLOGUES, mode
    views = ("  %xq_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x" + str(groups) + "x8xi32>\n"
             f"  %xm_k = index.constant {groups * 32} : offset\n"
             "  %xm_off = index.scale %tokens, %xm_k : index, offset -> offset\n"
             "  %xm_view = buffer.view %x[%xm_off] : buffer -> view<[%tokens]x" + str(groups) + "x2xf32>\n")
    new = (f"  %xf_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>\n"
           "  %pq_c127 = scalar.constant 127.0 : f32\n  %pq_ones8 = vector.constant 1 : vector<32xi8>\n")
    if mode == "silu":
        new += f"  %uf_view = buffer.view %u[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>\n"
    if mode == "norm":
        new += (f"  %nw_view = buffer.view %nw[%zero] : buffer -> view<{groups}x32xf32>\n"
                f"  %pn_count = scalar.constant {float(count)!r} : f32\n  %pn_eps = scalar.constant {eps!r} : f32\n")
    text = _one(text, views, new, "activation views")
    # the zero constants the quant needs exist in every iq kernel (bfly / iq setup)
    for c in ("  %zero_i = scalar.constant 0 : i32\n", "  %zero8i = vector.constant 0 : vector<8xi32>\n"):
        assert c in text, c
    blocks = []

    def sub(m):
        j = int(m.group(1))
        blocks.append(j)
        if mode == "f32":
            return _load(j, "xf_view", groups, "pv_x") + "\n" + quant_lines(j, f"pv_x{j}")
        if mode == "silu":
            return (_load(j, "xf_view", groups, "pv_f") + "\n" + _load(j, "uf_view", groups, "pv_u") + f"""
  %pv_sg{j} = vector.logisticf<afn> %pv_f{j} : vector<32xf32>
  %pv_s{j} = vector.mulf %pv_f{j}, %pv_sg{j} : vector<32xf32>
  %pv_o{j} = vector.mulf %pv_s{j}, %pv_u{j} : vector<32xf32>
""" + quant_lines(j, f"pv_o{j}"))
        # norm: g_post_add_norm's second half (the residual row is final: the producer accumulated into it)
        w = "" if blocks[:-1] else f"""  %pn_w = scf.if %ok -> (vector<32xf32>) {{
    %v = vector.load %nw_view[%g, %c0] : view<{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
"""
        return (_load(j, "xf_view", groups, "pv_x") + "\n" + w + f"""  %pn_sq{j} = vector.mulf %pv_x{j}, %pv_x{j} : vector<32xf32>
  %pn_lo{j} = vector.reduce<addf> %pn_sq{j}, %zero_f : vector<32xf32>, f32
  %pn_sum{j} = kernel.workgroup.reduce<addf> %pn_lo{j} : f32
  %pn_mean{j} = scalar.divf %pn_sum{j}, %pn_count : f32
  %pn_st{j} = scalar.addf %pn_mean{j}, %pn_eps : f32
  %pn_inv{j} = scalar.rsqrtf %pn_st{j} : f32
  %pn_invs{j} = vector.splat %pn_inv{j} : vector<32xf32>
  %pn_xn{j} = vector.mulf %pv_x{j}, %pn_invs{j} : vector<32xf32>
  %pn_h{j} = vector.mulf %pn_xn{j}, %pn_w : vector<32xf32>
""" + quant_lines(j, f"pn_h{j}"))
    text = TOKEN_BLOCK.sub(sub, text)
    assert blocks == sorted(blocks) and blocks and blocks[0] == 0, f"gen_pstep prologue: token blocks {blocks}"
    extra = {"silu": "%u: buffer, ", "norm": "%nw: buffer, "}.get(mode, "")
    if extra:
        text = _one(text, "%x: buffer, %w: buffer", "%x: buffer, " + extra + "%w: buffer", "launch args")
    return text


def pmv(text, prologue_mode="q8", mats=1, G=0, groups=None, eps=None, count=None):
    """The kernel text of gen_llm._mv (iq, one weight, no token chunks) -> its persistent / multi-matrix / prologue form."""
    name = re.search(r'kernel\.def export\("([^"]+)"\) @', text).group(1)
    m = re.match(r"g_mv_(\w+?)_k(\d+)_t(\d+)_q8_r(\d+)$", name)
    assert m, f"gen_pstep: {name} is not an int8 token-group matvec"
    fmt, K, TG, R = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    groups = groups or K // 32
    assert "%nch = index.constant" not in text and "%w2: buffer" not in text, f"{name}: chunked / two-weight kernels are not handled"
    new = f"g_pmv_{fmt}_k{K}_t{TG}_{prologue_mode}_r{R}" + (f"_m{mats}" if mats > 1 else "") + (f"_p{G}" if G else "")
    text = text.replace(f'kernel.def export("{name}") @{name}(', f'kernel.def export("{new}") @{new}(', 1)
    text = prologue(text, prologue_mode, groups, eps, count)
    # launch config: the row groups of every matrix, at most G workgroups
    cfg_old = "  %row_groups = index.div %n, %rows_per : index\n"
    cfg_new = cfg_old
    if mats == 2:
        text = _one(text, f"@{new}(%n: index, %tokens: index)", f"@{new}(%n: index, %n2: index, %tokens: index)", "config signature")
        cfg_new = ("  %rg1 = index.div %n, %rows_per : index\n  %rg2 = index.div %n2, %rows_per : index\n"
                   "  %row_groups = index.add %rg1, %rg2 : index\n")
    if G:
        cfg_new += f"  %cGp = index.constant {G} : index\n  %grid_rows = index.min %row_groups, %cGp : index\n"
        text = _one(text, "workgroups(%token_groups, %row_groups, %c1)", "workgroups(%token_groups, %grid_rows, %c1)", "grid")
    text = _one(text, cfg_old, cfg_new, "config row groups")
    if mats == 2:
        text = _one(text, "launch(%n_arg: index, %tokens_arg: index,", "launch(%n_arg: index, %n2_arg: index, %tokens_arg: index,", "launch signature")
        text = _one(text, "%w: buffer, %ws: buffer, %y: buffer)", "%w: buffer, %ws: buffer, %w2: buffer, %ws2: buffer, %y: buffer, %y2: buffer)", "weights")
        nas = re.search(r"  %n = index\.assume %n_arg \[([^\n]*)\] : index\n", text)
        text = text.replace(nas.group(0), nas.group(0) + "  %n2 = index.assume %n2_arg [" + nas.group(1).replace("%n_arg", "%n2_arg") + "] : index\n", 1)
    # the row work (row loop, partial fold barrier, stores) becomes the body of the row-group loop
    text = _one(text, ROW_BASE, "", "row base")
    head, rest = text.split(ROW_LOOP, 1)
    body, tail = rest.rsplit(TAIL, 1)
    assert not tail.strip(), f"{name}: text after the kernel"
    body = ROW_LOOP + body

    def ind(s):
        return "".join("  " + ln if ln.strip() else ln for ln in s.splitlines(True))
    loop = ("  %grid_y = index.div %n, %rows_per : index\n" if mats == 1 else
            "  %rg1 = index.div %n, %rows_per : index\n  %rg2 = index.div %n2, %rows_per : index\n  %grid_y = index.add %rg1, %rg2 : index\n")
    step = f"%cGs" if G else "%grid_y"
    if G:
        loop += f"  %cGs = index.constant {G} : index\n"
    loop += f"  scf.for %vg = [%group_id to %grid_y step {step}] {{\n"
    if mats == 1:
        loop += "    %row_base = index.mul %vg, %rows_per : index\n" + ind(body)
    else:
        b2 = body
        for a, b in (("@g_mxfp4s_icodes(%w, %ws, ", "@g_mxfp4s_icodes(%w2, %ws2, "), ("[lt(%row_value, %n)]", "[lt(%row_value, %n2)]"),
                     ("func.call @g_store_row(%y, %tokens, %n, ", "func.call @g_store_row(%y2, %tokens, %n2, ")):
            assert b2.count(a) == 1, f"{name}: second matrix anchor {a}"
            b2 = b2.replace(a, b)
        loop += ("    %in1 = index.cmp ult, %vg, %rg1 : index\n    scf.if %in1 {\n      %row_base = index.mul %vg, %rows_per : index\n"
                 + ind(ind(body)) + "    } else {\n      %vg2 = index.sub %vg, %rg1 : index\n      %row_base = index.mul %vg2, %rows_per : index\n"
                 + ind(ind(b2)) + "    }\n")
    # the next row group's wave partials overwrite the LDS the stores just read
    loop += "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n  }\n"
    return head + loop + TAIL
