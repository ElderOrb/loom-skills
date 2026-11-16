"""Producer-free MXFP4 prefill GEMM (lane pfa of campaign mxfp4st): the gfx12 fp8-WMMA GEMM of gen_mxfp4s whose PROLOGUE makes its own fp8 activations.

The fp8 GEMM of gen_mxfp4s.gemm_mxfp4s_f8_m256 reads activations that a producer kernel wrote (x8 + per-row scale s8). Here the GEMM reads the
f32 activation row itself and does the producer's work in the workgroup:
  1. stats pass (once per workgroup, before the K loop): thread t owns row t of the 256-row tile, reads the whole f32 row (K values), and
     computes its max |h| where h = x (plain: the row is already the GEMM input) or h = rmsnorm(x) * w (norm=True: the row stats pass also takes the
     sum of squares; the norm weight w is read from a buffer). From max |h| the per-row power-of-two scale 2^(7 - e) and the epilogue factor
     (the same F8_SCALE code as the producer twins: the row max lands in [128, 256), the weights' 2^8 shift is folded into the factor) follow;
  2. K loop: per 128-value step the thread re-reads its row's 128 values, forms h (x * w * rs * scale), converts to e4m3 and stores the row to LDS;
     the fragments, the weight decode, the MMA and the epilogue are those of the staged variant of gen_mxfp4s (a_direct=False).
No x8 / s8 buffers and no producer launch exist. A dead row (info live word 0: padding of an epoch) reads nothing and produces 0; a tile with no live
row does no work (the GEMM_SKIP_PADDING rule, decided from the info buffer here).
Cost model (why this is an alternative, not a default): every workgroup of a row tile repeats the stats pass and the conversion, so each activation
element is read and converted N / 128 times (N / 64 for the SwiGLU pair) instead of once; the f32 rows are 4x the bytes of the fp8 operand the GEMM
otherwise reads. Measured in kbench (suite gemm_mxfp4s_pfa) against the a_direct GEMM plus its producer.
Launch: launch(%rows, %k, %n, %x: buffer f32 [rows, k], %nw: buffer f32 [k] (norm weight; unused when norm=False), %info: buffer (token info, live word [row, 2]),
%w, %ws, [%w2, %w2s], %y f32 [rows, n])."""
from gen_mxfp4s import F8_SCALE


def gemm_mxfp4s_f8_pfa(mode, prefix="g", order="pairs", norm=False, kdim=5120, eps="1.0e-6", un=4, ag=2):
    """mode store (256 x 128 outputs per workgroup) or swiglu (256 x 64 of each of gate and up, silu(gate) * up). norm: the row is rmsnorm'ed (with
    the weight %nw, row length kdim) before the quantization. un: 16-value chunks per iteration of the stats loop; ag: chunks per group of the K loop's activation conversion."""
    two_w = mode == "swiglu"
    BM, NT, KS, PX, NB = 256, 256, 128, 144, 2
    TN = 64 if two_w else 128
    TW = TN * (2 if two_w else 1)
    RF = 2 if two_w else 4
    cfr = (8 if two_w else 16) // RF
    WC = TN // (16 * cfr)
    wave_cols = 16 * cfr
    name = f"{prefix}_gemm_mxfp4s_{mode}_f8_pfa{'_norm' if norm else ''}_rdna4"
    weights = ["w", "w2"] if two_w else ["w"]
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]
    L = []
    A = L.append
    A(f'''kernel.def target(@rdna4) export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant {BM} : index
  %cnt = index.constant {NT} : index
  %tn = index.constant {TN} : index
  %row_tiles = index.div %rows, %c256 : index
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%cnt, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %nw: buffer, %info: buffer, {", ".join(f"%{w}: buffer, %{w}s: buffer" for w in weights)}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, {BM}, 8192), multiple_of(%rows_arg, {BM})] : index
  %k = index.assume %k_arg [range(%k_arg, {KS}, 65536), multiple_of(%k_arg, {KS})] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %four = index.constant 4 : offset
  %one_off = index.constant 1 : offset
  %o0 = index.constant 0 : offset
  %o16 = index.constant 16 : offset
  %cj0 = index.constant 0 : index
  %cj1 = index.constant 1 : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %cks2 = index.constant {KS // 2} : index
  %cnb16 = index.constant {NB * 16} : index
  %cks32 = index.constant {KS // 32} : index
  %cnb = index.constant {NB} : index
  %c128 = index.constant {KS} : index
  %c256 = index.constant {BM} : index
  %tn = index.constant {TN} : index
  %row_tile_count = index.div %rows, %c256 : index
  %out_tile_count = index.div %n, %tn : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {NT - 1})] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, {NT // 32 - 1})] : index
  %cwc = index.constant {WC} : index
  %cwr = index.constant {16 * RF} : index
  %wave_row_group = index.div %bounded_subgroup, %cwc : index
  %wave_col_group = index.rem %bounded_subgroup, %cwc : index
  %wave_row_value = index.mul %wave_row_group, %cwr : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, {BM - 16 * RF})] : index''')
    for r in range(1, RF):
        A(f"  %wave_row{r}_c = index.constant {16 * r} : index\n  %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    A(f'''  %wave_cols = index.constant {wave_cols} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index
  %row_origin = index.mul %row_group, %c256 : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c32 : index
  %k_half = index.div %k, %c2 : index
  %k_half_off = index.scale %k_half, %one_off : index, offset -> offset
  %ks_off = index.scale %blocks, %one_off : index, offset -> offset
  %k16 = index.div %k, %c16 : index
  %xf_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k16]x16xf32>
  %nw_view = buffer.view %nw[%origin] : buffer -> view<[%k16]x16xf32>
  %live_view = buffer.view %info[%origin] : buffer -> view<[%rows]x4xi32>
  %x_bytes = index.constant {BM * PX} : offset
  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer
  %x_stage_q = buffer.view %x_stage[%origin] : buffer -> view<{BM}x{PX}xf8E4M3>
  %w_bytes = index.constant {TW * PX} : offset
  %w_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %w_stage_st = buffer.view %w_stage[%origin] : buffer -> view<{TW}x{PX}xi8>
  %a_schema = encoding.define #encoding.operand<element_format=f8e4m3, payload_elements=8, payload_registers=2> : encoding<schema>
  %b_schema = encoding.define #encoding.operand<element_format=f8e5m2, payload_elements=8, payload_registers=2> : encoding<schema>
  %a_layout = encoding.layout.dense : encoding<layout>
  %b_layout = encoding.layout.strided [1, {PX}] : encoding<layout>
  %a_storage = encoding.define #encoding.storage {{layout = %a_layout : encoding<layout>, schema = %a_schema : encoding<schema>}} : encoding<storage>
  %b_storage = encoding.define #encoding.storage {{layout = %b_layout : encoding<layout>, schema = %b_schema : encoding<schema>}} : encoding<storage>
  %x_stage_f8 = buffer.view %x_stage[%origin] : buffer -> view<{BM}x{PX}xf8E4M3, %a_storage>
  %w_stage_f8 = buffer.view %w_stage[%origin] : buffer -> view<{PX}x{TW}xf8E5M2, %b_storage>
  // the row of this thread (stats, quantization, epilogue factor) and whether it is live (not epoch padding)
  %x_row_value = index.add %row_origin, %workitem : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %live_word = view.load %live_view[%x_row, 2] : view<[%rows]x4xi32> -> i32
  %live_zero = scalar.constant 0 : i32
  %row_live = scalar.cmpi ne, %live_word, %live_zero : i32
  %live_f = scf.select %row_live, %one_f, %zero_f : f32
  %tile_any = kernel.workgroup.reduce<maxnumf> %live_f : f32
  %tile_live = scalar.cmpf one, %tile_any, %zero_f : f32
  scf.if %tile_live {{
  // W staging: thread t decodes two MXFP4 blocks (64 weights) of weight row t/2: columns (t%2) * 64
  %chalves = index.constant 2 : index
  %stage_row = index.div %workitem, %chalves : index
  %stage_half = index.rem %workitem, %chalves : index
  %stage_col = index.mul %stage_half, %cks2 : index''')
    if two_w:
        A(f'''  %half_tw = index.constant {TN} : index
  %w_is_up = index.cmp uge, %stage_row, %half_tw : index
  %w_local_up = index.sub %stage_row, %half_tw : index
  %w_local = scf.select %w_is_up, %w_local_up, %stage_row : index
  %w_out_row_value = index.add %out_origin, %w_local : index
  %wsel = scf.select %w_is_up, %w2, %w : buffer
  %wssel = scf.select %w_is_up, %w2s, %ws : buffer''')
    else:
        A("  %w_out_row_value = index.add %out_origin, %stage_row : index")
    A(f'''  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %k_half_off : index, offset -> offset
  %ws_row_origin = index.scale %w_out_row, %ks_off : index, offset -> offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c128 : index
  %k_steps_m1 = index.sub %k_steps, %c1 : index
  %c107 = scalar.constant 107 : i32
  %cs2 = scalar.constant 2 : i32
  %ones = scalar.constant 16843009 : i32
  %lo_mask = scalar.constant -256 : i32
  %tab_lo_c = scalar.constant 235669504 : i32
  %tab_hi_c = scalar.constant 370414096 : i32
  %m7 = vector.constant 7 : vector<4xi8>
  %m4 = vector.constant 4 : vector<4xi8>
  %m80 = vector.constant -128 : vector<4xi8>
  %m4i = vector.constant 4 : vector<1xi32>
  %m80i = vector.constant -2139062144 : vector<1xi32>''')
    for i in range(8):
        A(f"  %xo{i} = index.constant {16 * i} : index\n  %xi{i} = index.constant {i} : index")
    for u in range(max(un, ag)):
        A(f"  %su{u} = index.constant {u} : index")
    for q in range(ag):
        A(f"  %so{q} = index.constant {16 * q} : index")
    # ---- stats pass: max |h| (and the sum of squares for the norm) over the whole row ----
    ws = ssrc = None
    A(f"  %cun = index.constant {un} : index\n  %k_un = index.div %k16, %cun : index")
    carry = "%s0 = %zero_f : f32, %m0 = %zero_f : f32"
    A(f'''  %ss_row, %mx_row = scf.if %row_live -> (f32, f32) {{
    %ss_all, %mx_all = scf.for %c = [%c0 to %k_un step %c1]({carry}) -> (f32, f32) {{
      %cb = index.mul %c, %cun : index''')
    prev_s, prev_m = "%s0", "%m0"
    for u in range(un):
        A(f"      %ci{u}_v = index.add %cb, %su{u} : index\n      %ci{u} = index.assume %ci{u}_v [lt(%ci{u}_v, %k16)] : index")
        A(f"      %sv{u} = vector.load %xf_view[%x_row, %ci{u}, %c0] : view<[%rows]x[%k16]x16xf32> -> vector<16xf32>")
    for u in range(un):
        if norm:
            A(f"      %sw{u} = vector.load %nw_view[%ci{u}, %c0] : view<[%k16]x16xf32> -> vector<16xf32>\n      %sh{u} = vector.mulf %sv{u}, %sw{u} : vector<16xf32>")
            A(f"      %sq{u} = vector.mulf %sv{u}, %sv{u} : vector<16xf32>\n      %ssn{u} = vector.reduce<addf> %sq{u}, {prev_s} : vector<16xf32>, f32")
            prev_s = f"%ssn{u}"
            h = f"%sh{u}"
        else:
            h = f"%sv{u}"
        A(f"      %sa{u} = vector.absf {h} : vector<16xf32>\n      %smn{u} = vector.reduce<maxnumf> %sa{u}, {prev_m} : vector<16xf32>, f32")
        prev_m = f"%smn{u}"
    A(f'''      scf.yield {prev_s}, {prev_m} : f32, f32
    }}
    scf.yield %ss_all, %mx_all : f32, f32
  }} else {{
    scf.yield %zero_f, %zero_f : f32, f32
  }}''')
    # ---- row scale: the producer twins' exponent / F8_SCALE code ----
    if norm:
        A(f'''  %count = scalar.constant {float(kdim)} : f32
  %ms = scalar.divf %ss_row, %count : f32
  %eps = scalar.constant {eps} : f32
  %mst = scalar.addf %ms, %eps : f32
  %rs = scalar.rsqrtf %mst : f32
  %maximum = scalar.mulf %mx_row, %rs : f32''')
    else:
        A("  %maximum = scalar.addf %mx_row, %zero_f : f32")
    A(f'''  %bits = scalar.bitcast %maximum : f32 to i32
  %c23 = scalar.constant 23 : i32
  %c255 = scalar.constant 255 : i32
  %shifted = scalar.shrui %bits, %c23 : i32
  %biased = scalar.andi %shifted, %c255 : i32
  %scale = scalar.addf %live_f, %zero_f : f32
{F8_SCALE}''')
    A("  %mult = scalar.mulf %f8_scale, %rs : f32" if norm else "  %mult = scalar.addf %f8_scale, %zero_f : f32")

    wsrc = "%wsel" if two_w else "%w"
    ssrc = "%wssel" if two_w else "%ws"

    def loads(sfx, kstep):
        t = []
        t.append(f"    %wk{sfx}_a = index.mul {kstep}, %cks2 : index")
        t.append(f"    %wk{sfx}_b = index.mul %stage_half, %cnb16 : index")
        t.append(f"    %wk{sfx} = index.add %wk{sfx}_a, %wk{sfx}_b : index")
        t.append(f"    %wk{sfx}_o = index.scale %wk{sfx}, %one_off : index, offset -> offset")
        t.append(f"    %wkb{sfx} = index.add %w_row_origin, %wk{sfx}_o : offset")
        t.append(f"    %sb{sfx}_a = index.mul {kstep}, %cks32 : index")
        t.append(f"    %sb{sfx}_b = index.mul %stage_half, %cnb : index")
        t.append(f"    %sb{sfx} = index.add %sb{sfx}_a, %sb{sfx}_b : index")
        for j in range(NB):
            off = "%o0" if j == 0 else "%o16"
            t.append(f"    %wqo{j}{sfx} = index.add %wkb{sfx}, {off} : offset")
            t.append(f"    %wq{j}{sfx}_v = buffer.view {wsrc}[%wqo{j}{sfx}] : buffer -> view<4xi32>")
            t.append(f"    %wq{j}{sfx} = vector.load %wq{j}{sfx}_v[0] : view<4xi32> -> vector<4xi32>")
            t.append(f"    %sbj{j}{sfx} = index.add %sb{sfx}, %cj{j} : index")
            t.append(f"    %sbo{j}{sfx}_a = index.scale %sbj{j}{sfx}, %one_off : index, offset -> offset")
            t.append(f"    %sbo{j}{sfx} = index.add %ws_row_origin, %sbo{j}{sfx}_a : offset")
            t.append(f"    %we{j}{sfx}_v = buffer.view {ssrc}[%sbo{j}{sfx}] : buffer -> view<1xi8>")
            t.append(f"    %we{j}{sfx}_b = view.load %we{j}{sfx}_v[0] : view<1xi8> -> i8")
            t.append(f"    %we{j}{sfx} = scalar.extui %we{j}{sfx}_b : i8 to i32")
        return "\n".join(t)

    A(loads("p", "%c0"))

    def stage_w():
        t = []
        for j in range(NB):
            t.append(f'''    %em{j} = scalar.subi %we{j}c, %c107 : i32
    %ea{j} = scalar.shli %em{j}, %cs2 : i32
    %ea4_{j} = scalar.muli %ea{j}, %ones : i32
    %ea4z_{j} = scalar.andi %ea4_{j}, %lo_mask : i32
    %tlo{j} = scalar.addi %ea4z_{j}, %tab_lo_c : i32
    %thi{j} = scalar.addi %ea4_{j}, %tab_hi_c : i32
    %tab2_{j} = vector.from_elements %tlo{j}, %thi{j} : vector<2xi32>
    %tab{j} = vector.bitcast %tab2_{j} : vector<2xi32> to vector<8xi8>''')
            for i in range(4):
                t.append(f'''    %qw{j}{i}_1 = vector.slice %wq{j}c[{i}] : vector<4xi32> -> vector<1xi32>
    %qw{j}{i} = vector.bitcast %qw{j}{i}_1 : vector<1xi32> to vector<4xi8>
    %lo_i{j}{i} = vector.andi %qw{j}{i}, %m7 : vector<4xi8>
    %hs_i{j}{i} = vector.shrui %qw{j}{i}_1, %m4i : vector<1xi32>
    %hs_b{j}{i} = vector.bitcast %hs_i{j}{i} : vector<1xi32> to vector<4xi8>
    %hi_i{j}{i} = vector.andi %hs_b{j}{i}, %m7 : vector<4xi8>
    %lo_m{j}{i} = vector.table.lookup %tab{j}[%lo_i{j}{i}] : vector<8xi8>, vector<4xi8> -> vector<4xi8>
    %hi_m{j}{i} = vector.table.lookup %tab{j}[%hi_i{j}{i}] : vector<8xi8>, vector<4xi8> -> vector<4xi8>
    %ls_i{j}{i} = vector.shli %qw{j}{i}_1, %m4i : vector<1xi32>
    %ls_m{j}{i} = vector.andi %ls_i{j}{i}, %m80i : vector<1xi32>
    %ls_b{j}{i} = vector.bitcast %ls_m{j}{i} : vector<1xi32> to vector<4xi8>
    %hs_m{j}{i} = vector.andi %qw{j}{i}_1, %m80i : vector<1xi32>
    %hs_s{j}{i} = vector.bitcast %hs_m{j}{i} : vector<1xi32> to vector<4xi8>
    %lo_f{j}{i} = vector.ori %lo_m{j}{i}, %ls_b{j}{i} : vector<4xi8>
    %hi_f{j}{i} = vector.ori %hi_m{j}{i}, %hs_s{j}{i} : vector<4xi8>
    %lo_w{j}{i} = vector.bitcast %lo_f{j}{i} : vector<4xi8> to vector<1xi32>
    %hi_w{j}{i} = vector.bitcast %hi_f{j}{i} : vector<4xi8> to vector<1xi32>''')
            t.append(f'''    %lo4_{j} = vector.concat<0> %lo_w{j}0, %lo_w{j}1, %lo_w{j}2, %lo_w{j}3 : vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32> -> vector<4xi32>
    %hi4_{j} = vector.concat<0> %hi_w{j}0, %hi_w{j}1, %hi_w{j}2, %hi_w{j}3 : vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32> -> vector<4xi32>
    %lo16_{j} = vector.bitcast %lo4_{j} : vector<4xi32> to vector<16xi8>
    %hi16_{j} = vector.bitcast %hi4_{j} : vector<4xi32> to vector<16xi8>
    %wcl{j}_a = index.constant {32 * j} : index
    %wcl{j} = index.add %stage_col, %wcl{j}_a : index''')
            if order == "pairs":
                t.append(f'''    %blk{j} = vector.interleave<0> %lo16_{j}, %hi16_{j} : vector<16xi8>, vector<16xi8> -> vector<32xi8>
    vector.store %blk{j}, %w_stage_st[%stage_row, %wcl{j}] : vector<32xi8>, view<{TW}x{PX}xi8>''')
            else:
                t.append(f'''    %wch{j} = index.add %wcl{j}, %c16 : index
    vector.store %lo16_{j}, %w_stage_st[%stage_row, %wcl{j}] : vector<16xi8>, view<{TW}x{PX}xi8>
    vector.store %hi16_{j}, %w_stage_st[%stage_row, %wch{j}] : vector<16xi8>, view<{TW}x{PX}xi8>''')
        return "\n".join(t)

    carried = []
    for j in range(NB):
        carried.append((f"%we{j}c", f"%we{j}p", f"%we{j}n", "i32"))
        carried.append((f"%wq{j}c", f"%wq{j}p", f"%wq{j}n", "vector<4xi32>"))
    A("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) +
      " = scf.for %step = [%c0 to %k_steps step %c1](" +
      ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"{c[0]} = {c[1]} : {c[3]}" for c in carried) + ") -> (" +
      ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + ") {")
    # A: this thread's row, this step, `ag` 16-value chunks at a time: f32 loads, (norm) weights, convert, store to LDS. The weight decode of the step
    # sits between the first group's loads and its conversion so the loads fly under it.
    A("    %kb = index.mul %step, %c8 : index")
    A(stage_w())
    # the activation conversion as an inner loop of 8 // ag iterations (ag chunks each): it keeps the loads of the chunks from being hoisted all at once
    A(f"    %cag = index.constant {ag} : index\n    %cng = index.constant {8 // ag} : index\n    %cag16 = index.constant {16 * ag} : index")
    A("    scf.for %g = [%c0 to %cng step %c1] {")
    A("      %gc0 = index.mul %g, %cag : index\n      %gcol0 = index.mul %g, %cag16 : index")
    for q in range(ag):
        A(f"      %xc{q}_v0 = index.add %kb, %gc0 : index\n      %xc{q}_v = index.add %xc{q}_v0, %su{q} : index\n      %xc{q} = index.assume %xc{q}_v [lt(%xc{q}_v, %k16)] : index")
        A(f"      %xr{q} = vector.load %xf_view[%x_row, %xc{q}, %c0] : view<[%rows]x[%k16]x16xf32> -> vector<16xf32>")
        if norm:
            A(f"      %xw{q} = vector.load %nw_view[%xc{q}, %c0] : view<[%k16]x16xf32> -> vector<16xf32>")
    for q in range(ag):
        v = f"%xr{q}"
        if norm:
            A(f"      %xh{q} = vector.mulf %xr{q}, %xw{q} : vector<16xf32>")
            v = f"%xh{q}"
        A(f"      %mult16_{q} = vector.splat %mult : vector<16xf32>\n      %xs{q} = vector.mulf {v}, %mult16_{q} : vector<16xf32>\n      %xq{q} = vector.fptrunc %xs{q} : vector<16xf32> to vector<16xf8E4M3>")
        A(f"      %xoc{q} = index.add %gcol0, %so{q} : index")
        A(f"      vector.store %xq{q}, %x_stage_q[%workitem, %xoc{q}] : vector<16xf8E4M3>, view<{BM}x{PX}xf8E4M3>")
    A("    }")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    A("    %k_next_raw = index.add %step, %c1 : index\n    %k_next = index.min %k_next_raw, %k_steps_m1 : index")
    A(loads("n", "%k_next"))
    cur = {a: f"%i{a}" for a in accs}
    for kk in range(KS // 16):
        A(f"    %fk{kk} = index.constant {kk * 16} : index")
        for r in range(RF):
            A(f"    %l{r}_{kk} = vector.fragment.load<lhs> %x_stage_f8[{rowv[r]}, %fk{kk}] shape [%c16, %c16] : view<{BM}x{PX}xf8E4M3, %a_storage> -> vector<2xi32>")
        for wi in range(len(weights)):
            for c in range(cfr):
                A(f'''    %wr{wi}{c}_{kk}_c = index.constant {wi * TN + c * 16} : index
    %wr{wi}{c}_{kk} = index.add %wave_col, %wr{wi}{c}_{kk}_c : index
    %rf{wi}{c}_{kk} = vector.fragment.load<rhs> %w_stage_f8[%fk{kk}, %wr{wi}{c}_{kk}] shape [%c16, %c16] : view<{PX}x{TW}xf8E5M2, %b_storage> -> vector<2xi32>''')
        for wi in range(len(weights)):
            for c in range(cfr):
                for r in range(RF):
                    a = f"{wi}r{r}c{c}"
                    A(f"    %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<2xi32>, vector<2xi32>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    A("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(c[2] for c in carried) + " : " +
      ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + "\n  }")
    # epilogue: the row factor of this thread's row goes through LDS (the lane holding result row i needs row i's factor)
    A(f'''  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<{BM}x16xf32>
  %inverse16 = vector.splat %f8_inv : vector<16xf32>
  vector.store %inverse16, %inv_view[%workitem, %c0] : vector<16xf32>, view<{BM}x16xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)''')
    for r in range(RF):
        A(f"  %inv{r} = vector.fragment.load<result> %inv_view[{rowv[r]}, %c0] shape [%c16, %c16] : view<{BM}x16xf32> -> vector<8xf32>")
    A(f'''  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<{BM}x{TN}xf32, %y_layout>''')
    for r in range(RF):
        for c in range(cfr):
            A(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                A(f'''  %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
  %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
  %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                A(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>\n  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            A(f"  vector.fragment.store<result> %v{r}{c}, %y_view[{rowv[r]}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<{BM}x{TN}xf32, %y_layout>")
    A("  }\n  kernel.return\n}\n")
    return "\n".join(L)
