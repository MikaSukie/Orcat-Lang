; ModuleID = 'orcat'
source_filename = "main.orcat"
@__argv_ptr = global i8** null
declare i8* @malloc(i64)
declare void @free(i8*)
declare i64 @strlen(i8*)
declare i32 @puts(i8*)
declare void @exit(i32)
declare i64 @time(i64*)
declare void @srand(i32)
declare i32 @rand()
declare i32 @usleep(i32)

%struct.Person = type { i8*, i64 }

%enum.Opt = type { i32, [8 x i8] }
%enum.Option = type { i32, [8 x i8] }

@arrExample_gloarr1 = global [3 x i8*] zeroinitializer
@arrExample_gloarr1_len = global i32 3
@cash_item = global i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str0, i32 0, i32 0)
@cash_price = global double 0.00000000e+00
@cash_quantity = global i64 0
@cash_currency = global i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str1, i32 0, i32 0)
@cash_total = global double 0.00000000e+00
@orcat_argc_global = global i64 0
@orcat_argv_global = global i8** null

define i64 @orcat_argc() {
entry:
  %t0 = load i64, i64* @orcat_argc_global
  ret i64 %t0
}

define i8* @orcat_argv(i64 %idx) {
entry:
  %argvp = load i8**, i8*** @orcat_argv_global
  %isnull = icmp eq i8** %argvp, null
  br i1 %isnull, label %null_case, label %check_bounds
null_case:
  %src = getelementptr inbounds [5 x i8], [5 x i8]* @.str_null, i32 0, i32 0
  %alloc0 = call i8* @orcc_malloc(i64 5)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %alloc0, i8* %src, i64 5, i1 false)
  ret i8* %alloc0
check_bounds:
  %argc = load i64, i64* @orcat_argc_global
  %neg = icmp slt i64 %idx, 0
  %uge = icmp uge i64 %idx, %argc
  %oob = or i1 %neg, %uge
  br i1 %oob, label %null_case2, label %in_bounds
null_case2:
  %src2 = getelementptr inbounds [5 x i8], [5 x i8]* @.str_null, i32 0, i32 0
  %alloc1 = call i8* @orcc_malloc(i64 5)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %alloc1, i8* %src2, i64 5, i1 false)
  ret i8* %alloc1
in_bounds:
  %gep = getelementptr inbounds i8*, i8** %argvp, i64 %idx
  %val = load i8*, i8** %gep
  %len = call i64 @strlen(i8* %val)
  %allocsz = add i64 %len, 1
  %alloc2 = call i8* @orcc_malloc(i64 %allocsz)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %alloc2, i8* %val, i64 %allocsz, i1 false)
  ret i8* %alloc2
}

%async.test_async_inner = type { i32, i64 }
define %async.test_async_inner* @test_async_inner_init() {
entry:
  %szptr = getelementptr inbounds %async.test_async_inner, %async.test_async_inner* null, i32 1
  %sz = ptrtoint %async.test_async_inner* %szptr to i64
  %raw = call i8* @orcc_malloc(i64 %sz)
  %s = bitcast i8* %raw to %async.test_async_inner*
  %st0 = getelementptr inbounds %async.test_async_inner, %async.test_async_inner* %s, i32 0, i32 0
  store i32 0, i32* %st0
  ret %async.test_async_inner* %s
}
define i1 @test_async_inner_resume(%async.test_async_inner* %sm) {
entry:
  %stptr = getelementptr inbounds %async.test_async_inner, %async.test_async_inner* %sm, i32 0, i32 0
  %st = load i32, i32* %stptr
  switch i32 %st, label %state0 [
	i32 0, label %state0
  ]
state0:
  %f_addr = alloca i64
  store i64 0, i64* %f_addr
  %t1 = add i64 0, 25
  store i64 %t1, i64* %f_addr
  %t2 = load i64, i64* %f_addr
  %t3 = add i64 0, 50
  %t4 = add i64 %t2, %t3
  store i64 %t4, i64* %f_addr
  %t5 = load i64, i64* %f_addr
  %ret_ptr = getelementptr inbounds %async.test_async_inner, %async.test_async_inner* %sm, i32 0, i32 1
  store i64 %t5, i64* %ret_ptr
  ret i1 1
}

declare i64 @system(i8* %s)
declare i64 @ilength(i64 %integer)
declare i64 @flength(double %decimal)
declare i64 @slength(i8* %str)
declare i1 @streq(i8* %str1, i8* %str2)
declare i8* @get_os()
declare i8* @get_os_max_bits()
declare i8* @strtrim(i8* %in)
declare i8* @tal(i8* %s)
declare i8* @tac(i8* %s)
declare i8 @fcasti8(double %d)
declare i16 @fcasti16(double %d)
declare i32 @fcasti32(double %d)
declare i64 @fcasti64(double %d)
declare i64 @fcasti(double %d)
declare double @i8castf(i8 %i)
declare double @i16castf(i16 %i)
declare double @i32castf(i32 %i)
declare double @i64castf(i64 %i)
declare double @icastf(i64 %i)
declare i64 @char_at(i8* %s, i64 %idx)
declare i8* @sb_create()
declare i8* @sb_append_str(i8* %b, i8* %s)
declare i8* @sb_append_int(i8* %b, i64 %x)
declare i8* @sb_append_float(i8* %b, double %f)
declare i8* @sb_append_bool(i8* %b, i1 %bb)
declare i8* @sb_append_float32(i8* %b, float %f)
declare i8* @sb_finish(i8* %b)
declare i8* @itostr(i64 %i)
declare i8* @i8tostr(i8 %i)
declare i8* @i16tostr(i16 %i)
declare i8* @i32tostr(i32 %i)
declare i8* @i64tostr(i64 %i)
declare i8* @f32tostr(float %f)
declare i8* @ftostr(double %f)
declare i8* @btostr(i1 %b)
declare i8* @tostr(i8* %s)
define i8* @safestring(i8* %s) {
entry:
  %s_addr = alloca i8*
  store i8* %s, i8** %s_addr
  %t6 = call i8* @sb_create()
  %t7 = load i8*, i8** %s_addr
  %t8 = call i64 @strlen(i8* %t6)
  %t9 = call i64 @strlen(i8* %t7)
  %t10 = add i64 %t8, %t9
  %t11 = add i64 %t10, 1
  %t12 = call i8* @orcc_malloc(i64 %t11)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t12, i8* %t6, i64 %t8, i1 false)
  %t13 = getelementptr inbounds i8, i8* %t12, i64 %t8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t13, i8* %t7, i64 %t9, i1 false)
  %t14 = getelementptr inbounds i8, i8* %t12, i64 %t10
  store i8 0, i8* %t14
  ret i8* %t12
}
declare double @tofloat(i64 %x)
declare i64 @toint(double %x)
declare i64 @toint_round(double %x)
declare i8* @rmtrz(double %x)
declare i1 @contains(i8* %var, i8* %detection_char)
declare i64 @countcontain(i8* %x, i8* %detection_char)
define i8* @empty() {
entry:
  %t15 = getelementptr inbounds [1 x i8], [1 x i8]* @.str3, i32 0, i32 0
  %t16 = call i8* @tostr(i8* %t15)
  ret i8* %t16
}
declare i8* @input(i8* %input)
declare i64 @iinput(i8* %input)
declare i1 @binput(i8* %input)
declare double @finput(i8* %input)
declare i8* @sinput(i8* %input)
declare void @print(i8* %s)
define void @printnl(i8* %s) {
entry:
  %s_addr = alloca i8*
  store i8* %s, i8** %s_addr
  %t17 = load i8*, i8** %s_addr
  call void @print(i8* %t17)
  call void @newline()
  ret void
}
define void @newline() {
entry:
  %t18 = getelementptr inbounds [2 x i8], [2 x i8]* @.str4, i32 0, i32 0
  call void @print(i8* %t18)
  ret void
}
declare i8* @read_file(i8* %path)
declare i1 @write_file(i8* %path, i8* %content)
declare i1 @append_file(i8* %path, i8* %content)
declare i1 @file_exists(i8* %path)
declare i8* @read_lines(i8* %path)
declare void @free_str(i8* %s)
define i8* @try_read_file(i8* %path) {
entry:
  %path_addr = alloca i8*
  store i8* %path, i8** %path_addr
  %t19 = load i8*, i8** %path_addr
  %t20 = call i1 @file_exists(i8* %t19)
  br i1 %t20, label %then1, label %else2
then1:
  %t21 = load i8*, i8** %path_addr
  %t22 = call i8* @read_file(i8* %t21)
  ret i8* %t22
else2:
  %t23 = getelementptr inbounds [1 x i8], [1 x i8]* @.str5, i32 0, i32 0
  ret i8* %t23
endif3:
  ret i8* null
}
define i1 @save_text(i8* %path, i8* %text) {
entry:
  %path_addr = alloca i8*
  store i8* %path, i8** %path_addr
  %text_addr = alloca i8*
  store i8* %text, i8** %text_addr
  %t24 = load i8*, i8** %path_addr
  %t25 = load i8*, i8** %text_addr
  %t26 = call i1 @write_file(i8* %t24, i8* %t25)
  ret i1 %t26
}
define i1 @add_text(i8* %path, i8* %text) {
entry:
  %path_addr = alloca i8*
  store i8* %path, i8** %path_addr
  %text_addr = alloca i8*
  store i8* %text, i8** %text_addr
  %t27 = load i8*, i8** %path_addr
  %t28 = load i8*, i8** %text_addr
  %t29 = call i1 @append_file(i8* %t27, i8* %t28)
  ret i1 %t29
}
declare i64 @Umake_int(i64 %x)
declare i64 @Umake_float(double %f)
declare i64 @Umake_bool(i1 %b)
declare i64 @Umake_string(i8* %s)
declare i64 @Uget_tag(i64 %h)
declare i64 @Uget_int(i64 %h)
declare double @Uget_float(i64 %h)
declare i1 @Uget_bool(i64 %h)
declare i8* @Uget_string(i64 %h)
declare void @Ufree_union(i64 %h)
define i64 @test_nottest() {
entry:
  %b_addr = alloca i1
  store i1 0, i1* %b_addr
  %t30 = add i1 0, 1
  store i1 %t30, i1* %b_addr
  %t31 = load i1, i1* %b_addr
  %t32 = load i1, i1* %b_addr
  %t33 = xor i1 %t32, true
  br i1 %t33, label %then4, label %else5
then4:
  %t34 = getelementptr inbounds [18 x i8], [18 x i8]* @.str6, i32 0, i32 0
  call void @print(i8* %t34)
  br label %endif6
else5:
  %t35 = getelementptr inbounds [8 x i8], [8 x i8]* @.str7, i32 0, i32 0
  call void @print(i8* %t35)
  br label %endif6
endif6:
  %t36 = add i64 0, 0
  ret i64 %t36
}
define i64 @test_args() {
entry:
  %argc_addr = alloca i64
  store i64 0, i64* %argc_addr
  %t37 = call i64 @orcat_argc()
  store i64 %t37, i64* %argc_addr
  %t38 = load i64, i64* %argc_addr
  %t39 = call i8* @itostr(i64 %t38)
  %t40 = getelementptr inbounds [2 x i8], [2 x i8]* @.str8, i32 0, i32 0
  %t41 = call i64 @strlen(i8* %t39)
  %t42 = call i64 @strlen(i8* %t40)
  %t43 = add i64 %t41, %t42
  %t44 = add i64 %t43, 1
  %t45 = call i8* @orcc_malloc(i64 %t44)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t45, i8* %t39, i64 %t41, i1 false)
  %t46 = getelementptr inbounds i8, i8* %t45, i64 %t41
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t46, i8* %t40, i64 %t42, i1 false)
  %t47 = getelementptr inbounds i8, i8* %t45, i64 %t43
  store i8 0, i8* %t47
  call void @print(i8* %t45)
  %t48 = getelementptr inbounds [7 x i8], [7 x i8]* @.str9, i32 0, i32 0
  %t49 = call i8* @tostr(i8* %t48)
  %t50 = add i64 0, 1
  %t51 = call i8* @orcat_argv(i64 %t50)
  %t52 = call i8* @tostr(i8* %t51)
  %t53 = call i64 @strlen(i8* %t49)
  %t54 = call i64 @strlen(i8* %t52)
  %t55 = add i64 %t53, %t54
  %t56 = add i64 %t55, 1
  %t57 = call i8* @orcc_malloc(i64 %t56)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t57, i8* %t49, i64 %t53, i1 false)
  %t58 = getelementptr inbounds i8, i8* %t57, i64 %t53
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t58, i8* %t52, i64 %t54, i1 false)
  %t59 = getelementptr inbounds i8, i8* %t57, i64 %t55
  store i8 0, i8* %t59
  call void @print(i8* %t57)
  call void @newline()
  %t60 = add i64 0, 0
  ret i64 %t60
}
define i64 @test_arrayExample() {
entry:
  %arr1_addr = alloca [7 x i8*]
  store [7 x i8*] zeroinitializer, [7 x i8*]* %arr1_addr
  %arr1_len  = alloca i32
  store i32 7, i32* %arr1_len
  %t61 = add i64 0, 2
  %t62 = getelementptr inbounds [5 x i8], [5 x i8]* @.str10, i32 0, i32 0
  %t63 = trunc i64 %t61 to i32
  %t64 = load i32, i32* %arr1_len
  %t65 = icmp ult i32 %t63, %t64
  br i1 %t65, label %oob_ok8, label %oob_fail7
oob_fail7:
  call void @orcc_oob_abort()
  unreachable
oob_ok8:
  %t66 = getelementptr inbounds [7 x i8*], [7 x i8*]* %arr1_addr, i32 0, i32 %t63
  store i8* %t62, i8** %t66
  %t67 = add i64 0, 2
  %t70 = trunc i64 %t67 to i32
  %t71 = load i32, i32* %arr1_len
  %t72 = icmp ult i32 %t70, %t71
  br i1 %t72, label %oob_ok10, label %oob_fail9
oob_fail9:
  call void @orcc_oob_abort()
  unreachable
oob_ok10:
  %t68 = getelementptr inbounds [7 x i8*], [7 x i8*]* %arr1_addr, i32 0, i32 %t70
  %t69 = load i8*, i8** %t68
  call void @print(i8* %t69)
  call void @newline()
  %t73 = add i64 0, 1
  %t74 = getelementptr inbounds [7 x i8], [7 x i8]* @.str11, i32 0, i32 0
  %t75 = trunc i64 %t73 to i32
  %t76 = load i32, i32* @arrExample_gloarr1_len
  %t77 = icmp ult i32 %t75, %t76
  br i1 %t77, label %oob_ok12, label %oob_fail11
oob_fail11:
  call void @orcc_oob_abort()
  unreachable
oob_ok12:
  %t78 = getelementptr inbounds [3 x i8*], [3 x i8*]* @arrExample_gloarr1, i32 0, i32 %t75
  store i8* %t74, i8** %t78
  %t79 = add i64 0, 1
  %t82 = trunc i64 %t79 to i32
  %t83 = load i32, i32* @arrExample_gloarr1_len
  %t84 = icmp ult i32 %t82, %t83
  br i1 %t84, label %oob_ok14, label %oob_fail13
oob_fail13:
  call void @orcc_oob_abort()
  unreachable
oob_ok14:
  %t80 = getelementptr inbounds [3 x i8*], [3 x i8*]* @arrExample_gloarr1, i32 0, i32 %t82
  %t81 = load i8*, i8** %t80
  call void @print(i8* %t81)
  %t85 = add i64 0, 0
  ret i64 %t85
}
define void @cash_process() {
entry:
  %possesive_addr = alloca i8*
  store i8* null, i8** %possesive_addr
  %t86 = getelementptr inbounds [35 x i8], [35 x i8]* @.str12, i32 0, i32 0
  %t87 = call i8* @sinput(i8* %t86)
  store i8* %t87, i8** @cash_item
  %t88 = getelementptr inbounds [30 x i8], [30 x i8]* @.str13, i32 0, i32 0
  %t89 = call double @finput(i8* %t88)
  store double %t89, double* @cash_price
  %t90 = getelementptr inbounds [27 x i8], [27 x i8]* @.str14, i32 0, i32 0
  %t91 = call i64 @iinput(i8* %t90)
  store i64 %t91, i64* @cash_quantity
  %t92 = load double, double* @cash_price
  %t93 = load i64, i64* @cash_quantity
  %t94 = call double @tofloat(i64 %t93)
  %t95 = fmul double %t92, %t94
  store double %t95, double* @cash_total
  %t96 = getelementptr inbounds [1 x i8], [1 x i8]* @.str15, i32 0, i32 0
  store i8* %t96, i8** %possesive_addr
  %t97 = load i64, i64* @cash_quantity
  %t98 = add i64 0, 1
  %t99 = icmp sgt i64 %t97, %t98
  br i1 %t99, label %then15, label %endif16
then15:
  %t100 = getelementptr inbounds [2 x i8], [2 x i8]* @.str16, i32 0, i32 0
  store i8* %t100, i8** %possesive_addr
  br label %endif16
endif16:
  %t101 = getelementptr inbounds [17 x i8], [17 x i8]* @.str17, i32 0, i32 0
  %t102 = call i8* @tostr(i8* %t101)
  %t103 = load i64, i64* @cash_quantity
  %t104 = call i8* @itostr(i64 %t103)
  %t105 = call i64 @strlen(i8* %t102)
  %t106 = call i64 @strlen(i8* %t104)
  %t107 = add i64 %t105, %t106
  %t108 = add i64 %t107, 1
  %t109 = call i8* @orcc_malloc(i64 %t108)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t109, i8* %t102, i64 %t105, i1 false)
  %t110 = getelementptr inbounds i8, i8* %t109, i64 %t105
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t110, i8* %t104, i64 %t106, i1 false)
  %t111 = getelementptr inbounds i8, i8* %t109, i64 %t107
  store i8 0, i8* %t111
  %t112 = getelementptr inbounds [2 x i8], [2 x i8]* @.str18, i32 0, i32 0
  %t113 = call i64 @strlen(i8* %t109)
  %t114 = call i64 @strlen(i8* %t112)
  %t115 = add i64 %t113, %t114
  %t116 = add i64 %t115, 1
  %t117 = call i8* @orcc_malloc(i64 %t116)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t117, i8* %t109, i64 %t113, i1 false)
  %t118 = getelementptr inbounds i8, i8* %t117, i64 %t113
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t118, i8* %t112, i64 %t114, i1 false)
  %t119 = getelementptr inbounds i8, i8* %t117, i64 %t115
  store i8 0, i8* %t119
  %t120 = load i8*, i8** @cash_item
  %t121 = call i64 @strlen(i8* %t117)
  %t122 = call i64 @strlen(i8* %t120)
  %t123 = add i64 %t121, %t122
  %t124 = add i64 %t123, 1
  %t125 = call i8* @orcc_malloc(i64 %t124)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t125, i8* %t117, i64 %t121, i1 false)
  %t126 = getelementptr inbounds i8, i8* %t125, i64 %t121
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t126, i8* %t120, i64 %t122, i1 false)
  %t127 = getelementptr inbounds i8, i8* %t125, i64 %t123
  store i8 0, i8* %t127
  %t128 = load i8*, i8** %possesive_addr
  %t129 = call i64 @strlen(i8* %t125)
  %t130 = call i64 @strlen(i8* %t128)
  %t131 = add i64 %t129, %t130
  %t132 = add i64 %t131, 1
  %t133 = call i8* @orcc_malloc(i64 %t132)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t133, i8* %t125, i64 %t129, i1 false)
  %t134 = getelementptr inbounds i8, i8* %t133, i64 %t129
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t134, i8* %t128, i64 %t130, i1 false)
  %t135 = getelementptr inbounds i8, i8* %t133, i64 %t131
  store i8 0, i8* %t135
  %t136 = getelementptr inbounds [15 x i8], [15 x i8]* @.str19, i32 0, i32 0
  %t137 = call i64 @strlen(i8* %t133)
  %t138 = call i64 @strlen(i8* %t136)
  %t139 = add i64 %t137, %t138
  %t140 = add i64 %t139, 1
  %t141 = call i8* @orcc_malloc(i64 %t140)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t141, i8* %t133, i64 %t137, i1 false)
  %t142 = getelementptr inbounds i8, i8* %t141, i64 %t137
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t142, i8* %t136, i64 %t138, i1 false)
  %t143 = getelementptr inbounds i8, i8* %t141, i64 %t139
  store i8 0, i8* %t143
  %t144 = load i8*, i8** @cash_currency
  %t145 = call i64 @strlen(i8* %t141)
  %t146 = call i64 @strlen(i8* %t144)
  %t147 = add i64 %t145, %t146
  %t148 = add i64 %t147, 1
  %t149 = call i8* @orcc_malloc(i64 %t148)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t149, i8* %t141, i64 %t145, i1 false)
  %t150 = getelementptr inbounds i8, i8* %t149, i64 %t145
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t150, i8* %t144, i64 %t146, i1 false)
  %t151 = getelementptr inbounds i8, i8* %t149, i64 %t147
  store i8 0, i8* %t151
  %t152 = load double, double* @cash_total
  %t153 = call i8* @rmtrz(double %t152)
  %t154 = call i64 @strlen(i8* %t149)
  %t155 = call i64 @strlen(i8* %t153)
  %t156 = add i64 %t154, %t155
  %t157 = add i64 %t156, 1
  %t158 = call i8* @orcc_malloc(i64 %t157)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t158, i8* %t149, i64 %t154, i1 false)
  %t159 = getelementptr inbounds i8, i8* %t158, i64 %t154
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t159, i8* %t153, i64 %t155, i1 false)
  %t160 = getelementptr inbounds i8, i8* %t158, i64 %t156
  store i8 0, i8* %t160
  %t161 = getelementptr inbounds [2 x i8], [2 x i8]* @.str20, i32 0, i32 0
  %t162 = call i64 @strlen(i8* %t158)
  %t163 = call i64 @strlen(i8* %t161)
  %t164 = add i64 %t162, %t163
  %t165 = add i64 %t164, 1
  %t166 = call i8* @orcc_malloc(i64 %t165)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t166, i8* %t158, i64 %t162, i1 false)
  %t167 = getelementptr inbounds i8, i8* %t166, i64 %t162
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t167, i8* %t161, i64 %t163, i1 false)
  %t168 = getelementptr inbounds i8, i8* %t166, i64 %t164
  store i8 0, i8* %t168
  call void @print(i8* %t166)
  ret void
}
define i64 @test_cashregisterexample() {
entry:
  call void @cash_process()
  %t169 = add i64 0, 0
  ret i64 %t169
}
define i64 @test_continueandbreaktest() {
entry:
  %i_addr = alloca i64
  store i64 0, i64* %i_addr
  %t170 = add i64 0, 0
  store i64 %t170, i64* %i_addr
  br label %while_head17
while_head17:
  %t171 = load i64, i64* %i_addr
  %t172 = add i64 0, 5
  %t173 = icmp slt i64 %t171, %t172
  br i1 %t173, label %while_body18, label %while_end19
while_body18:
  %t174 = load i64, i64* %i_addr
  %t175 = add i64 0, 1
  %t176 = add i64 %t174, %t175
  store i64 %t176, i64* %i_addr
  %t177 = load i64, i64* %i_addr
  %t178 = add i64 0, 3
  %t179 = icmp eq i64 %t177, %t178
  br i1 %t179, label %then20, label %endif21
then20:
  %t180 = getelementptr inbounds [10 x i8], [10 x i8]* @.str21, i32 0, i32 0
  call void @print(i8* %t180)
  br label %while_head17
  unreachable
endif21:
  %t181 = load i64, i64* %i_addr
  %t182 = add i64 0, 4
  %t183 = icmp eq i64 %t181, %t182
  br i1 %t183, label %then22, label %endif23
then22:
  %t184 = getelementptr inbounds [10 x i8], [10 x i8]* @.str22, i32 0, i32 0
  call void @print(i8* %t184)
  br label %while_end19
  unreachable
endif23:
  %t185 = getelementptr inbounds [6 x i8], [6 x i8]* @.str23, i32 0, i32 0
  call void @print(i8* %t185)
  %t186 = load i64, i64* %i_addr
  %t187 = call i8* @itostr(i64 %t186)
  call void @print(i8* %t187)
  %t188 = getelementptr inbounds [2 x i8], [2 x i8]* @.str24, i32 0, i32 0
  call void @print(i8* %t188)
  br label %while_head17
while_end19:
  %t189 = add i64 0, 0
  ret i64 %t189
}
define i64 @test_filemain_inner() {
entry:
  %path_addr = alloca i8*
  store i8* null, i8** %path_addr
  %msg_addr = alloca i8*
  store i8* null, i8** %msg_addr
  %ok_addr = alloca i1
  store i1 0, i1* %ok_addr
  %data_addr = alloca i8*
  store i8* null, i8** %data_addr
  %t190 = getelementptr inbounds [12 x i8], [12 x i8]* @.str25, i32 0, i32 0
  store i8* %t190, i8** %path_addr
  %t191 = getelementptr inbounds [14 x i8], [14 x i8]* @.str26, i32 0, i32 0
  store i8* %t191, i8** %msg_addr
  %t192 = load i8*, i8** %path_addr
  %t193 = load i8*, i8** %msg_addr
  %t194 = call i1 @save_text(i8* %t192, i8* %t193)
  store i1 %t194, i1* %ok_addr
  %t195 = load i1, i1* %ok_addr
  br i1 %t195, label %then24, label %endif25
then24:
  %t196 = getelementptr inbounds [15 x i8], [15 x i8]* @.str27, i32 0, i32 0
  call void @print(i8* %t196)
  br label %endif25
endif25:
  %t197 = load i8*, i8** %path_addr
  %t198 = call i8* @read_lines(i8* %t197)
  store i8* %t198, i8** %data_addr
  %t199 = load i8*, i8** %data_addr
  call void @print(i8* %t199)
  %t200 = load i8*, i8** %data_addr
  call void @free_str(i8* %t200)
  %t201 = add i64 0, 0
  ret i64 %t201
}
define i64 @test_filemain() {
entry:
  %t202 = call i64 @test_filemain_inner()
  ret i64 %t202
}
define void @greet_person(%struct.Person* %p) {
entry:
  %p_addr = alloca %struct.Person*
  store %struct.Person* %p, %struct.Person** %p_addr
  %t203 = getelementptr inbounds [8 x i8], [8 x i8]* @.str28, i32 0, i32 0
  call void @print(i8* %t203)
  %t204 = load %struct.Person*, %struct.Person** %p_addr
  %t205 = icmp eq %struct.Person* %t204, null
  br i1 %t205, label %null_ptr_fail26, label %null_ptr_ok27
null_ptr_fail26:
  call void @orcc_null_abort()
  unreachable
null_ptr_ok27:
  %t206 = getelementptr inbounds %struct.Person, %struct.Person* %t204, i32 0, i32 0
  %t207 = load i8*, i8** %t206
  call void @print(i8* %t207)
  %t208 = getelementptr inbounds [3 x i8], [3 x i8]* @.str29, i32 0, i32 0
  call void @print(i8* %t208)
  ret void
}
define i64 @test_generictypesandstructs() {
entry:
  %p_addr = alloca %struct.Person*
  store %struct.Person* null, %struct.Person** %p_addr
  %t209 = alloca %struct.Person
  %t210 = getelementptr inbounds [6 x i8], [6 x i8]* @.str30, i32 0, i32 0
  %t211 = getelementptr inbounds %struct.Person, %struct.Person* %t209, i32 0, i32 0
  store i8* %t210, i8** %t211
  %t212 = add i64 0, 30
  %t213 = getelementptr inbounds %struct.Person, %struct.Person* %t209, i32 0, i32 1
  store i64 %t212, i64* %t213
  store %struct.Person* %t209, %struct.Person** %p_addr
  %t214 = load %struct.Person*, %struct.Person** %p_addr
  call void @greet_person(%struct.Person* %t214)
  %t215 = add i64 0, 0
  ret i64 %t215
}
define i64 @test_getosandbits() {
entry:
  %t216 = call i8* @get_os()
  call void @print(i8* %t216)
  call void @newline()
  %t217 = call i8* @get_os_max_bits()
  call void @print(i8* %t217)
  call void @newline()
  %t218 = add i64 0, 0
  ret i64 %t218
}
define i64 @test_inputexample_inner() {
entry:
  %res_addr = alloca i64
  store i64 0, i64* %res_addr
  %t219 = getelementptr inbounds [20 x i8], [20 x i8]* @.str31, i32 0, i32 0
  %t220 = call i64 @iinput(i8* %t219)
  store i64 %t220, i64* %res_addr
  %t221 = getelementptr inbounds [8 x i8], [8 x i8]* @.str32, i32 0, i32 0
  %t222 = call i8* @tostr(i8* %t221)
  %t223 = load i64, i64* %res_addr
  %t224 = call i8* @itostr(i64 %t223)
  %t225 = call i64 @strlen(i8* %t222)
  %t226 = call i64 @strlen(i8* %t224)
  %t227 = add i64 %t225, %t226
  %t228 = add i64 %t227, 1
  %t229 = call i8* @orcc_malloc(i64 %t228)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t229, i8* %t222, i64 %t225, i1 false)
  %t230 = getelementptr inbounds i8, i8* %t229, i64 %t225
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t230, i8* %t224, i64 %t226, i1 false)
  %t231 = getelementptr inbounds i8, i8* %t229, i64 %t227
  store i8 0, i8* %t231
  %t232 = getelementptr inbounds [13 x i8], [13 x i8]* @.str33, i32 0, i32 0
  %t233 = call i8* @tostr(i8* %t232)
  %t234 = call i64 @strlen(i8* %t229)
  %t235 = call i64 @strlen(i8* %t233)
  %t236 = add i64 %t234, %t235
  %t237 = add i64 %t236, 1
  %t238 = call i8* @orcc_malloc(i64 %t237)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t238, i8* %t229, i64 %t234, i1 false)
  %t239 = getelementptr inbounds i8, i8* %t238, i64 %t234
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t239, i8* %t233, i64 %t235, i1 false)
  %t240 = getelementptr inbounds i8, i8* %t238, i64 %t236
  store i8 0, i8* %t240
  call void @print(i8* %t238)
  %t241 = add i64 0, 0
  ret i64 %t241
}
define i64 @test_inputexample() {
entry:
  %t242 = call i64 @test_inputexample_inner()
  ret i64 %t242
}
define i64 @test_mathoptest() {
entry:
  %x_addr = alloca i64
  store i64 0, i64* %x_addr
  %y_addr = alloca i64
  store i64 0, i64* %y_addr
  %z_addr = alloca i64
  store i64 0, i64* %z_addr
  %t243 = add i64 0, 6
  store i64 %t243, i64* %x_addr
  %t244 = load i64, i64* %x_addr
  %t245 = add i64 0, 4
  %t246 = add i64 %t244, %t245
  store i64 %t246, i64* %x_addr
  %t247 = getelementptr inbounds [9 x i8], [9 x i8]* @.str34, i32 0, i32 0
  call void @print(i8* %t247)
  %t248 = load i64, i64* %x_addr
  %t249 = call i8* @itostr(i64 %t248)
  call void @print(i8* %t249)
  %t250 = getelementptr inbounds [2 x i8], [2 x i8]* @.str35, i32 0, i32 0
  call void @print(i8* %t250)
  %t251 = load i64, i64* %x_addr
  %t252 = add i64 0, 2
  %t253 = sub i64 %t251, %t252
  store i64 %t253, i64* %x_addr
  %t254 = getelementptr inbounds [9 x i8], [9 x i8]* @.str36, i32 0, i32 0
  call void @print(i8* %t254)
  %t255 = load i64, i64* %x_addr
  %t256 = call i8* @itostr(i64 %t255)
  call void @print(i8* %t256)
  %t257 = getelementptr inbounds [2 x i8], [2 x i8]* @.str37, i32 0, i32 0
  call void @print(i8* %t257)
  %t258 = load i64, i64* %x_addr
  %t259 = add i64 0, 3
  %t260 = mul i64 %t258, %t259
  store i64 %t260, i64* %x_addr
  %t261 = getelementptr inbounds [9 x i8], [9 x i8]* @.str38, i32 0, i32 0
  call void @print(i8* %t261)
  %t262 = load i64, i64* %x_addr
  %t263 = call i8* @itostr(i64 %t262)
  call void @print(i8* %t263)
  %t264 = getelementptr inbounds [2 x i8], [2 x i8]* @.str39, i32 0, i32 0
  call void @print(i8* %t264)
  %t265 = load i64, i64* %x_addr
  %t266 = add i64 0, 2
  %t267 = sdiv i64 %t265, %t266
  store i64 %t267, i64* %x_addr
  %t268 = getelementptr inbounds [9 x i8], [9 x i8]* @.str40, i32 0, i32 0
  call void @print(i8* %t268)
  %t269 = load i64, i64* %x_addr
  %t270 = call i8* @itostr(i64 %t269)
  call void @print(i8* %t270)
  %t271 = getelementptr inbounds [2 x i8], [2 x i8]* @.str41, i32 0, i32 0
  call void @print(i8* %t271)
  %t272 = load i64, i64* %x_addr
  %t273 = add i64 0, 5
  %t274 = srem i64 %t272, %t273
  store i64 %t274, i64* %x_addr
  %t275 = getelementptr inbounds [9 x i8], [9 x i8]* @.str42, i32 0, i32 0
  call void @print(i8* %t275)
  %t276 = load i64, i64* %x_addr
  %t277 = call i8* @itostr(i64 %t276)
  call void @print(i8* %t277)
  %t278 = getelementptr inbounds [2 x i8], [2 x i8]* @.str43, i32 0, i32 0
  call void @print(i8* %t278)
  %t279 = load i64, i64* %x_addr
  %t280 = add i64 0, 3
  %t281 = and i64 %t279, %t280
  store i64 %t281, i64* %x_addr
  %t282 = getelementptr inbounds [9 x i8], [9 x i8]* @.str44, i32 0, i32 0
  call void @print(i8* %t282)
  %t283 = load i64, i64* %x_addr
  %t284 = call i8* @itostr(i64 %t283)
  call void @print(i8* %t284)
  %t285 = getelementptr inbounds [2 x i8], [2 x i8]* @.str45, i32 0, i32 0
  call void @print(i8* %t285)
  %t286 = load i64, i64* %x_addr
  %t287 = add i64 0, 8
  %t288 = or i64 %t286, %t287
  store i64 %t288, i64* %x_addr
  %t289 = getelementptr inbounds [9 x i8], [9 x i8]* @.str46, i32 0, i32 0
  call void @print(i8* %t289)
  %t290 = load i64, i64* %x_addr
  %t291 = call i8* @itostr(i64 %t290)
  call void @print(i8* %t291)
  %t292 = getelementptr inbounds [2 x i8], [2 x i8]* @.str47, i32 0, i32 0
  call void @print(i8* %t292)
  %t293 = load i64, i64* %x_addr
  %t294 = add i64 0, 1
  %t295 = xor i64 %t293, %t294
  store i64 %t295, i64* %x_addr
  %t296 = getelementptr inbounds [9 x i8], [9 x i8]* @.str48, i32 0, i32 0
  call void @print(i8* %t296)
  %t297 = load i64, i64* %x_addr
  %t298 = call i8* @itostr(i64 %t297)
  call void @print(i8* %t298)
  %t299 = getelementptr inbounds [2 x i8], [2 x i8]* @.str49, i32 0, i32 0
  call void @print(i8* %t299)
  %t300 = load i64, i64* %x_addr
  %t301 = add i64 0, 2
  %t302 = shl i64 %t300, %t301
  store i64 %t302, i64* %x_addr
  %t303 = getelementptr inbounds [10 x i8], [10 x i8]* @.str50, i32 0, i32 0
  call void @print(i8* %t303)
  %t304 = load i64, i64* %x_addr
  %t305 = call i8* @itostr(i64 %t304)
  call void @print(i8* %t305)
  %t306 = getelementptr inbounds [2 x i8], [2 x i8]* @.str51, i32 0, i32 0
  call void @print(i8* %t306)
  %t307 = load i64, i64* %x_addr
  %t308 = add i64 0, 1
  %t309 = ashr i64 %t307, %t308
  store i64 %t309, i64* %x_addr
  %t310 = getelementptr inbounds [10 x i8], [10 x i8]* @.str52, i32 0, i32 0
  call void @print(i8* %t310)
  %t311 = load i64, i64* %x_addr
  %t312 = call i8* @itostr(i64 %t311)
  call void @print(i8* %t312)
  %t313 = getelementptr inbounds [2 x i8], [2 x i8]* @.str53, i32 0, i32 0
  call void @print(i8* %t313)
  %t314 = add i64 0, 6
  store i64 %t314, i64* %x_addr
  %t315 = load i64, i64* %x_addr
  %t316 = add i64 0, 3
  %t317 = and i64 %t315, %t316
  %t318 = add i64 0, 8
  %t319 = or i64 %t317, %t318
  store i64 %t319, i64* %y_addr
  %t320 = load i64, i64* %y_addr
  %t321 = add i64 0, 1
  %t322 = ashr i64 %t320, %t321
  store i64 %t322, i64* %z_addr
  %t323 = getelementptr inbounds [14 x i8], [14 x i8]* @.str54, i32 0, i32 0
  call void @print(i8* %t323)
  %t324 = load i64, i64* %y_addr
  %t325 = call i8* @itostr(i64 %t324)
  call void @print(i8* %t325)
  %t326 = getelementptr inbounds [2 x i8], [2 x i8]* @.str55, i32 0, i32 0
  call void @print(i8* %t326)
  %t327 = getelementptr inbounds [9 x i8], [9 x i8]* @.str56, i32 0, i32 0
  call void @print(i8* %t327)
  %t328 = load i64, i64* %z_addr
  %t329 = call i8* @itostr(i64 %t328)
  call void @print(i8* %t329)
  %t330 = getelementptr inbounds [2 x i8], [2 x i8]* @.str57, i32 0, i32 0
  call void @print(i8* %t330)
  %t331 = load i64, i64* %x_addr
  ret i64 %t331
}
define void @allocate_and_free_test() {
entry:
  %ptr1_addr = alloca i64*
  store i64* null, i64** %ptr1_addr
  %ptr2_addr = alloca i64*
  store i64* null, i64** %ptr2_addr
  %ptr3_addr = alloca i64*
  store i64* null, i64** %ptr3_addr
  %ptr4_addr = alloca i64*
  store i64* null, i64** %ptr4_addr
  %ptr5_addr = alloca i64*
  store i64* null, i64** %ptr5_addr
  %nested1_addr = alloca i64*
  store i64* null, i64** %nested1_addr
  %nested2_addr = alloca i64*
  store i64* null, i64** %nested2_addr
  %ptr6_addr = alloca i64*
  store i64* null, i64** %ptr6_addr
  %ptr7_addr = alloca i64*
  store i64* null, i64** %ptr7_addr
  %ptrx_addr = alloca i64*
  store i64* null, i64** %ptrx_addr
  %ptry_addr = alloca i64*
  store i64* null, i64** %ptry_addr
  %ptr1_t332_addr = alloca i64*
  store i64* null, i64** %ptr1_t332_addr
  %t333 = add i64 0, 8
  %t334 = call i8* @orcc_malloc(i64 %t333)
  store i64* %t334, i64** %ptr1_t332_addr
  %ptr2_t336_addr = alloca i64*
  store i64* null, i64** %ptr2_t336_addr
  %t337 = add i64 0, 8
  %t338 = call i8* @orcc_malloc(i64 %t337)
  store i64* %t338, i64** %ptr2_t336_addr
  %ptr3_t340_addr = alloca i64*
  store i64* null, i64** %ptr3_t340_addr
  %t341 = add i64 0, 8
  %t342 = call i8* @orcc_malloc(i64 %t341)
  store i64* %t342, i64** %ptr3_t340_addr
  %ptr4_t344_addr = alloca i64*
  store i64* null, i64** %ptr4_t344_addr
  %t345 = add i64 0, 8
  %t346 = call i8* @orcc_malloc(i64 %t345)
  store i64* %t346, i64** %ptr4_t344_addr
  %ptr5_t348_addr = alloca i64*
  store i64* null, i64** %ptr5_t348_addr
  %t349 = add i64 0, 8
  %t350 = call i8* @orcc_malloc(i64 %t349)
  store i64* %t350, i64** %ptr5_t348_addr
  %t352 = load i64*, i64** %ptr1_t332_addr
  %t353 = add i64 0, 1
  store i64 %t353, i64* %t352
  %t354 = load i64*, i64** %ptr2_t336_addr
  %t355 = add i64 0, 2
  store i64 %t355, i64* %t354
  %t356 = load i64*, i64** %ptr3_t340_addr
  %t357 = add i64 0, 3
  store i64 %t357, i64* %t356
  %t358 = getelementptr inbounds [37 x i8], [37 x i8]* @.str58, i32 0, i32 0
  call void @print(i8* %t358)
  call void @newline()
  %nested1_t359_addr = alloca i64*
  store i64* null, i64** %nested1_t359_addr
  %t360 = add i64 0, 8
  %t361 = call i8* @orcc_malloc(i64 %t360)
  store i64* %t361, i64** %nested1_t359_addr
  %nested2_t363_addr = alloca i64*
  store i64* null, i64** %nested2_t363_addr
  %t364 = add i64 0, 8
  %t365 = call i8* @orcc_malloc(i64 %t364)
  store i64* %t365, i64** %nested2_t363_addr
  %t367 = load i64*, i64** %nested1_t359_addr
  %t368 = add i64 0, 100
  store i64 %t368, i64* %t367
  %t369 = load i64*, i64** %nested2_t363_addr
  %t370 = add i64 0, 200
  store i64 %t370, i64* %t369
  %t371 = getelementptr inbounds [25 x i8], [25 x i8]* @.str59, i32 0, i32 0
  call void @print(i8* %t371)
  call void @newline()
  %t372 = load i64*, i64** %nested1_t359_addr
  %t373 = bitcast i64* %t372 to i8*
  call void @orcc_free(i8* %t373)
  store i64* null, i64** %nested1_t359_addr
  %t374 = load i64*, i64** %nested2_t363_addr
  %t375 = bitcast i64* %t374 to i8*
  call void @orcc_free(i8* %t375)
  store i64* null, i64** %nested2_t363_addr
  %t376 = getelementptr inbounds [26 x i8], [26 x i8]* @.str60, i32 0, i32 0
  call void @print(i8* %t376)
  call void @newline()
  %ptr6_t377_addr = alloca i64*
  store i64* null, i64** %ptr6_t377_addr
  %t378 = add i64 0, 8
  %t379 = call i8* @orcc_malloc(i64 %t378)
  store i64* %t379, i64** %ptr6_t377_addr
  %ptr7_t381_addr = alloca i64*
  store i64* null, i64** %ptr7_t381_addr
  %t382 = add i64 0, 8
  %t383 = call i8* @orcc_malloc(i64 %t382)
  store i64* %t383, i64** %ptr7_t381_addr
  %t385 = load i64*, i64** %ptr6_t377_addr
  %t386 = add i64 0, 6
  store i64 %t386, i64* %t385
  %t387 = load i64*, i64** %ptr7_t381_addr
  %t388 = add i64 0, 7
  store i64 %t388, i64* %t387
  %t389 = getelementptr inbounds [31 x i8], [31 x i8]* @.str61, i32 0, i32 0
  call void @print(i8* %t389)
  call void @newline()
  %t390 = load i64*, i64** %ptr1_t332_addr
  %t391 = bitcast i64* %t390 to i8*
  call void @orcc_free(i8* %t391)
  store i64* null, i64** %ptr1_t332_addr
  %t392 = load i64*, i64** %ptr2_t336_addr
  %t393 = bitcast i64* %t392 to i8*
  call void @orcc_free(i8* %t393)
  store i64* null, i64** %ptr2_t336_addr
  %t394 = load i64*, i64** %ptr3_t340_addr
  %t395 = bitcast i64* %t394 to i8*
  call void @orcc_free(i8* %t395)
  store i64* null, i64** %ptr3_t340_addr
  %t396 = load i64*, i64** %ptr4_t344_addr
  %t397 = bitcast i64* %t396 to i8*
  call void @orcc_free(i8* %t397)
  store i64* null, i64** %ptr4_t344_addr
  %t398 = load i64*, i64** %ptr5_t348_addr
  %t399 = bitcast i64* %t398 to i8*
  call void @orcc_free(i8* %t399)
  store i64* null, i64** %ptr5_t348_addr
  %t400 = load i64*, i64** %ptr6_t377_addr
  %t401 = bitcast i64* %t400 to i8*
  call void @orcc_free(i8* %t401)
  store i64* null, i64** %ptr6_t377_addr
  %t402 = load i64*, i64** %ptr7_t381_addr
  %t403 = bitcast i64* %t402 to i8*
  call void @orcc_free(i8* %t403)
  store i64* null, i64** %ptr7_t381_addr
  %t404 = getelementptr inbounds [29 x i8], [29 x i8]* @.str62, i32 0, i32 0
  call void @print(i8* %t404)
  call void @newline()
  %t405 = getelementptr inbounds [29 x i8], [29 x i8]* @.str63, i32 0, i32 0
  call void @print(i8* %t405)
  call void @newline()
  %t406 = add i64 0, 8
  %t407 = call i8* @orcc_malloc(i64 %t406)
  store i64* %t407, i64** %ptrx_addr
  %t408 = getelementptr inbounds [14 x i8], [14 x i8]* @.str64, i32 0, i32 0
  call void @print(i8* %t408)
  %t409 = add i64 0, 8
  %t410 = call i8* @orcc_malloc(i64 %t409)
  store i64* %t410, i64** %ptry_addr
  %t412 = load i64*, i64** %ptry_addr
  %t413 = bitcast i64* %t412 to i8*
  call void @orcc_free(i8* %t413)
  store i64* null, i64** %ptry_addr
  call void @newline()
  ret void
}
define i64 @test_memtest() {
entry:
  call void @allocate_and_free_test()
  %t414 = add i64 0, 0
  ret i64 %t414
}
define i64 @test_C_RuntimeUnionTest() {
entry:
  %a_addr = alloca i64
  store i64 0, i64* %a_addr
  %b_addr = alloca i64
  store i64 0, i64* %b_addr
  %c_addr = alloca i64
  store i64 0, i64* %c_addr
  %d_addr = alloca i64
  store i64 0, i64* %d_addr
  %sa_addr = alloca i8*
  store i8* null, i8** %sa_addr
  %sb_addr = alloca i8*
  store i8* null, i8** %sb_addr
  %sc_addr = alloca i8*
  store i8* null, i8** %sc_addr
  %t415 = add i64 0, 123
  %t416 = call i64 @Umake_int(i64 %t415)
  store i64 %t416, i64* %a_addr
  %t417 = fadd double 0.0, 3.14000000e+00
  %t418 = call i64 @Umake_float(double %t417)
  store i64 %t418, i64* %b_addr
  %t419 = add i1 0, 1
  %t420 = call i64 @Umake_bool(i1 %t419)
  store i64 %t420, i64* %c_addr
  %t421 = getelementptr inbounds [16 x i8], [16 x i8]* @.str65, i32 0, i32 0
  %t422 = call i64 @Umake_string(i8* %t421)
  store i64 %t422, i64* %d_addr
  %t423 = load i64, i64* %a_addr
  %t424 = call i64 @Uget_tag(i64 %t423)
  %t425 = add i64 0, 1
  %t426 = icmp eq i64 %t424, %t425
  br i1 %t426, label %then28, label %endif29
then28:
  %sa_t427_addr = alloca i8*
  store i8* null, i8** %sa_t427_addr
  %t428 = load i64, i64* %a_addr
  %t429 = call i64 @Uget_int(i64 %t428)
  %t430 = call i8* @itostr(i64 %t429)
  store i8* %t430, i8** %sa_t427_addr
  %t431 = getelementptr inbounds [11 x i8], [11 x i8]* @.str66, i32 0, i32 0
  call void @print(i8* %t431)
  %t432 = load i8*, i8** %sa_t427_addr
  call void @print(i8* %t432)
  call void @newline()
  %t433 = load i8*, i8** %sa_t427_addr
  call void @free_str(i8* %t433)
  br label %endif29
endif29:
  %t434 = load i64, i64* %b_addr
  %t435 = call i64 @Uget_tag(i64 %t434)
  %t436 = add i64 0, 2
  %t437 = icmp eq i64 %t435, %t436
  br i1 %t437, label %then30, label %endif31
then30:
  %sb_t438_addr = alloca i8*
  store i8* null, i8** %sb_t438_addr
  %t439 = load i64, i64* %b_addr
  %t440 = call double @Uget_float(i64 %t439)
  %t441 = call i8* @ftostr(double %t440)
  store i8* %t441, i8** %sb_t438_addr
  %t442 = getelementptr inbounds [13 x i8], [13 x i8]* @.str67, i32 0, i32 0
  call void @print(i8* %t442)
  %t443 = load i8*, i8** %sb_t438_addr
  call void @print(i8* %t443)
  call void @newline()
  %t444 = load i8*, i8** %sb_t438_addr
  call void @free_str(i8* %t444)
  br label %endif31
endif31:
  %t445 = load i64, i64* %c_addr
  %t446 = call i64 @Uget_tag(i64 %t445)
  %t447 = add i64 0, 3
  %t448 = icmp eq i64 %t446, %t447
  br i1 %t448, label %then32, label %endif33
then32:
  %sc_t449_addr = alloca i8*
  store i8* null, i8** %sc_t449_addr
  %t450 = load i64, i64* %c_addr
  %t451 = call i1 @Uget_bool(i64 %t450)
  %t452 = call i8* @btostr(i1 %t451)
  store i8* %t452, i8** %sc_t449_addr
  %t453 = getelementptr inbounds [12 x i8], [12 x i8]* @.str68, i32 0, i32 0
  call void @print(i8* %t453)
  %t454 = load i8*, i8** %sc_t449_addr
  call void @print(i8* %t454)
  call void @newline()
  %t455 = load i8*, i8** %sc_t449_addr
  call void @free_str(i8* %t455)
  br label %endif33
endif33:
  %t456 = load i64, i64* %d_addr
  %t457 = call i64 @Uget_tag(i64 %t456)
  %t458 = add i64 0, 4
  %t459 = icmp eq i64 %t457, %t458
  br i1 %t459, label %then34, label %endif35
then34:
  %t460 = getelementptr inbounds [14 x i8], [14 x i8]* @.str69, i32 0, i32 0
  call void @print(i8* %t460)
  %t461 = load i64, i64* %d_addr
  %t462 = call i8* @Uget_string(i64 %t461)
  call void @print(i8* %t462)
  call void @newline()
  br label %endif35
endif35:
  %t463 = load i64, i64* %a_addr
  call void @Ufree_union(i64 %t463)
  %t464 = load i64, i64* %b_addr
  call void @Ufree_union(i64 %t464)
  %t465 = load i64, i64* %c_addr
  call void @Ufree_union(i64 %t465)
  %t466 = load i64, i64* %d_addr
  call void @Ufree_union(i64 %t466)
  %t467 = add i64 0, 0
  ret i64 %t467
}
define i64 @test_async() {
entry:
  %t468 = call %async.test_async_inner* @test_async_inner_init()
  %t469 = call i1 @test_async_inner_resume(%async.test_async_inner* %t468)
  br i1 %t469, label %await_cont36, label %await_suspend37
await_suspend37:
  %t470 = bitcast i1 (%async.test_async_inner*)* @test_async_inner_resume to i8*
  %t471 = bitcast %async.test_async_inner* %t468 to i8*
  call void @orcc_register_async(i8* %t470, i8* %t471)
  call void @orcc_block_until_complete(i8* %t471)
  br label %await_cont36
await_cont36:
  %t472 = getelementptr inbounds %async.test_async_inner, %async.test_async_inner* %t468, i32 0, i32 1
  %t473 = load i64, i64* %t472
  ret i64 %t473
}
define i64 @test_cast() {
entry:
  %t474 = getelementptr inbounds [5 x i8], [5 x i8]* @.str70, i32 0, i32 0
  call void @print(i8* %t474)
  call void @newline()
  %t475 = add i64 0, 0
  ret i64 %t475
}
define i64 @test_enum1() {
entry:
  %c_addr = alloca i64
  store i64 0, i64* %c_addr
  %t476 = add i64 0, 1
  store i64 %t476, i64* %c_addr
  %t477 = load i64, i64* %c_addr
  switch i64 %t477, label %match_end38 [
	i64 0, label %case_Red39
	i64 1, label %case_Green40
	i64 2, label %case_Blue41
  ]
case_Red39:
  %t478 = add i64 0, 0
  ret i64 %t478
case_Green40:
  %t479 = add i64 0, 1
  ret i64 %t479
case_Blue41:
  %t480 = add i64 0, 2
  ret i64 %t480
match_end38:
  %t481 = add i64 0, 0
  ret i64 %t481
}
define i64 @test_enum2() {
entry:
  %o_addr = alloca %enum.Opt*
  store %enum.Opt* null, %enum.Opt** %o_addr
  %t482 = add i64 0, 42
  %t483 = getelementptr inbounds %enum.Opt, %enum.Opt* null, i32 1
  %t484 = ptrtoint %enum.Opt* %t483 to i64
  %t485 = call i8* @orcc_malloc(i64 %t484)
  %t486 = bitcast i8* %t485 to %enum.Opt*
  %t487 = getelementptr inbounds %enum.Opt, %enum.Opt* %t486, i32 0, i32 0
  store i32 1, i32* %t487
  %t488 = getelementptr inbounds %enum.Opt, %enum.Opt* %t486, i32 0, i32 1
  store i64 %t482, i64* %t488
  store %enum.Opt* %t486, %enum.Opt** %o_addr
  %t490 = load %enum.Opt*, %enum.Opt** %o_addr
  %t491 = getelementptr inbounds %enum.Opt, %enum.Opt* %t490, i32 0, i32 0
  %t492 = load i32, i32* %t491
  switch i32 %t492, label %match_end42 [
	i32 0, label %case_None43
	i32 1, label %case_Some44
  ]
case_None43:
  %t493 = add i64 0, 0
  ret i64 %t493
case_Some44:
  %t494 = getelementptr inbounds %enum.Opt, %enum.Opt* %t490, i32 0, i32 1
  %t495 = load i64, i64* %t494
  %x_addr = alloca i64
  store i64 %t495, i64* %x_addr
  %t496 = load i64, i64* %x_addr
  ret i64 %t496
match_end42:
  %t497 = add i64 0, 0
  ret i64 %t497
}
define i64 @test_generic_enum() {
entry:
  %t498 = add i64 0, 0
  ret i64 %t498
}
define i64 @test_get_address() {
entry:
  %x_addr = alloca i64
  store i64 0, i64* %x_addr
  %p_addr = alloca i64*
  store i64* null, i64** %p_addr
  %t499 = add i64 0, 42
  store i64 %t499, i64* %x_addr
  store i64* %x_addr, i64** %p_addr
  %t500 = load i64*, i64** %p_addr
  %t501 = add i64 0, 100
  store i64 %t501, i64* %t500
  %t502 = getelementptr inbounds [2 x i8], [2 x i8]* @.str71, i32 0, i32 0
  call void @print(i8* %t502)
  %t503 = add i64 0, 0
  ret i64 %t503
}
define void @pair_int_float(i64 %a, double %b) {
entry:
  %a_addr = alloca i64
  store i64 %a, i64* %a_addr
  %b_addr = alloca double
  store double %b, double* %b_addr
  ret void
}
define i64 @test_multigeneric() {
entry:
  %t504 = add i64 0, 1
  %t505 = fadd double 0.0, 2.00000000e+00
  call void @pair_int_float(i64 %t504, double %t505)
  %t506 = add i64 0, 0
  ret i64 %t506
}
define i8* @tostring_int(i64 %candidate) {
entry:
  %candidate_addr = alloca i64
  store i64 %candidate, i64* %candidate_addr
  %typcand_addr = alloca i8*
  store i8* null, i8** %typcand_addr
  %t552 = getelementptr inbounds [4 x i8], [4 x i8]* @.str92, i32 0, i32 0
  store i8* %t552, i8** %typcand_addr
  %t553 = load i8*, i8** %typcand_addr
  %t554 = getelementptr inbounds [4 x i8], [4 x i8]* @.str93, i32 0, i32 0
  %t555 = call i1 @streq(i8* %t553, i8* %t554)
  br i1 %t555, label %then45, label %endif46
then45:
  %t556 = call i8* @get_os_max_bits()
  %t557 = getelementptr inbounds [3 x i8], [3 x i8]* @.str94, i32 0, i32 0
  %t558 = call i1 @streq(i8* %t556, i8* %t557)
  br i1 %t558, label %then47, label %else48
then47:
  %t559 = load i64, i64* %candidate_addr
  %t560 = call i8* @i64tostr(i64 %t559)
  ret i8* %t560
else48:
  %t561 = call i8* @get_os_max_bits()
  %t562 = getelementptr inbounds [3 x i8], [3 x i8]* @.str95, i32 0, i32 0
  %t563 = call i1 @streq(i8* %t561, i8* %t562)
  br i1 %t563, label %then50, label %else51
then50:
  %t564 = load i64, i64* %candidate_addr
  %t565 = trunc i64 %t564 to i32
  %t566 = call i8* @i32tostr(i32 %t565)
  ret i8* %t566
else51:
  %t567 = load i64, i64* %candidate_addr
  %t568 = call i8* @itostr(i64 %t567)
  ret i8* %t568
endif52:
  br label %endif49
endif49:
  br label %endif46
endif46:
  %t569 = load i8*, i8** %typcand_addr
  %t570 = getelementptr inbounds [6 x i8], [6 x i8]* @.str96, i32 0, i32 0
  %t571 = call i1 @streq(i8* %t569, i8* %t570)
  br i1 %t571, label %then53, label %endif54
then53:
  %t572 = load i64, i64* %candidate_addr
  %t573 = sitofp i64 %t572 to double
  %t574 = call i8* @ftostr(double %t573)
  ret i8* %t574
endif54:
  %t575 = load i8*, i8** %typcand_addr
  %t576 = getelementptr inbounds [5 x i8], [5 x i8]* @.str97, i32 0, i32 0
  %t577 = call i1 @streq(i8* %t575, i8* %t576)
  br i1 %t577, label %then55, label %endif56
then55:
  %t578 = load i64, i64* %candidate_addr
  %t579 = trunc i64 %t578 to i1
  %t580 = call i8* @btostr(i1 %t579)
  ret i8* %t580
endif56:
  %t581 = load i8*, i8** %typcand_addr
  %t582 = getelementptr inbounds [7 x i8], [7 x i8]* @.str98, i32 0, i32 0
  %t583 = call i1 @streq(i8* %t581, i8* %t582)
  br i1 %t583, label %then57, label %endif58
then57:
  %t584 = load i64, i64* %candidate_addr
  %t585 = inttoptr i64 %t584 to i8*
  %t586 = call i8* @tostr(i8* %t585)
  ret i8* %t586
endif58:
  %t587 = getelementptr inbounds [6 x i8], [6 x i8]* @.str99, i32 0, i32 0
  ret i8* %t587
}
define i64 @user_main() {
entry:
  %t507 = getelementptr inbounds [23 x i8], [23 x i8]* @.str72, i32 0, i32 0
  call void @print(i8* %t507)
  %t508 = call i64 @test_nottest()
  call void @newline()
  %t509 = getelementptr inbounds [20 x i8], [20 x i8]* @.str73, i32 0, i32 0
  call void @print(i8* %t509)
  %t510 = call i64 @test_args()
  call void @newline()
  %t511 = getelementptr inbounds [28 x i8], [28 x i8]* @.str74, i32 0, i32 0
  call void @print(i8* %t511)
  %t512 = call i64 @test_arrayExample()
  call void @newline()
  %t513 = getelementptr inbounds [35 x i8], [35 x i8]* @.str75, i32 0, i32 0
  call void @print(i8* %t513)
  %t514 = call i64 @test_cashregisterexample()
  call void @newline()
  %t515 = getelementptr inbounds [36 x i8], [36 x i8]* @.str76, i32 0, i32 0
  call void @print(i8* %t515)
  %t516 = call i64 @test_continueandbreaktest()
  call void @newline()
  %t517 = getelementptr inbounds [24 x i8], [24 x i8]* @.str77, i32 0, i32 0
  call void @print(i8* %t517)
  %t518 = call i64 @test_filemain()
  call void @newline()
  %t519 = getelementptr inbounds [38 x i8], [38 x i8]* @.str78, i32 0, i32 0
  call void @print(i8* %t519)
  %t520 = call i64 @test_generictypesandstructs()
  call void @newline()
  %t521 = getelementptr inbounds [28 x i8], [28 x i8]* @.str79, i32 0, i32 0
  call void @print(i8* %t521)
  %t522 = call i64 @test_getosandbits()
  call void @newline()
  %t523 = getelementptr inbounds [28 x i8], [28 x i8]* @.str80, i32 0, i32 0
  call void @print(i8* %t523)
  %t524 = call i64 @test_inputexample()
  call void @newline()
  %t525 = getelementptr inbounds [26 x i8], [26 x i8]* @.str81, i32 0, i32 0
  call void @print(i8* %t525)
  %t526 = call i64 @test_mathoptest()
  call void @newline()
  %t527 = getelementptr inbounds [23 x i8], [23 x i8]* @.str82, i32 0, i32 0
  call void @print(i8* %t527)
  %t528 = call i64 @test_memtest()
  call void @newline()
  %t529 = getelementptr inbounds [34 x i8], [34 x i8]* @.str83, i32 0, i32 0
  call void @print(i8* %t529)
  %t530 = call i64 @test_C_RuntimeUnionTest()
  call void @newline()
  %t531 = getelementptr inbounds [21 x i8], [21 x i8]* @.str84, i32 0, i32 0
  call void @print(i8* %t531)
  %t532 = call i64 @test_async()
  %t533 = call i8* @itostr(i64 %t532)
  call void @print(i8* %t533)
  call void @newline()
  %t534 = getelementptr inbounds [20 x i8], [20 x i8]* @.str85, i32 0, i32 0
  call void @print(i8* %t534)
  %t535 = call i64 @test_cast()
  call void @newline()
  %t536 = getelementptr inbounds [21 x i8], [21 x i8]* @.str86, i32 0, i32 0
  call void @print(i8* %t536)
  %t537 = call i64 @test_enum1()
  %t538 = call i8* @itostr(i64 %t537)
  call void @print(i8* %t538)
  call void @newline()
  %t539 = getelementptr inbounds [21 x i8], [21 x i8]* @.str87, i32 0, i32 0
  call void @print(i8* %t539)
  %t540 = call i64 @test_enum2()
  %t541 = call i8* @itostr(i64 %t540)
  call void @print(i8* %t541)
  call void @newline()
  %t542 = getelementptr inbounds [28 x i8], [28 x i8]* @.str88, i32 0, i32 0
  call void @print(i8* %t542)
  %t543 = call i64 @test_generic_enum()
  %t544 = call i8* @itostr(i64 %t543)
  call void @print(i8* %t544)
  call void @newline()
  %t545 = getelementptr inbounds [27 x i8], [27 x i8]* @.str89, i32 0, i32 0
  call void @print(i8* %t545)
  %t546 = call i64 @test_get_address()
  call void @newline()
  %t547 = getelementptr inbounds [28 x i8], [28 x i8]* @.str90, i32 0, i32 0
  call void @print(i8* %t547)
  %t548 = call i64 @test_multigeneric()
  %t549 = call i8* @itostr(i64 %t548)
  call void @print(i8* %t549)
  call void @newline()
  %t550 = getelementptr inbounds [28 x i8], [28 x i8]* @.str91, i32 0, i32 0
  call void @print(i8* %t550)
  %t551 = add i64 0, 0
  %t588 = call i8* @tostring_int(i64 %t551)
  call void @print(i8* %t588)
  call void @newline()
  %t589 = getelementptr inbounds [28 x i8], [28 x i8]* @.str100, i32 0, i32 0
  call void @print(i8* %t589)
  %t590 = add i64 0, 0
  ret i64 %t590
}
@.str0 = private unnamed_addr constant [1 x i8] c"\00"
@.str1 = private unnamed_addr constant [2 x i8] c"$\00"
@.str_null = private unnamed_addr constant [5 x i8] c"null\00"
@.str3 = private unnamed_addr constant [1 x i8] c"\00"
@.str4 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str5 = private unnamed_addr constant [1 x i8] c"\00"
@.str6 = private unnamed_addr constant [18 x i8] c"should not print\0A\00"
@.str7 = private unnamed_addr constant [8 x i8] c"works!\0A\00"
@.str8 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str9 = private unnamed_addr constant [7 x i8] c"args: \00"
@.str10 = private unnamed_addr constant [5 x i8] c"test\00"
@.str11 = private unnamed_addr constant [7 x i8] c"testar\00"
@.str12 = private unnamed_addr constant [35 x i8] c"What item would you like to buy?: \00"
@.str13 = private unnamed_addr constant [30 x i8] c"What is the price for each?: \00"
@.str14 = private unnamed_addr constant [27 x i8] c"How many would you like?: \00"
@.str15 = private unnamed_addr constant [1 x i8] c"\00"
@.str16 = private unnamed_addr constant [2 x i8] c"s\00"
@.str17 = private unnamed_addr constant [17 x i8] c"You have bought \00"
@.str18 = private unnamed_addr constant [2 x i8] c" \00"
@.str19 = private unnamed_addr constant [15 x i8] c" which costed \00"
@.str20 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str21 = private unnamed_addr constant [10 x i8] c"SKIPPING\0A\00"
@.str22 = private unnamed_addr constant [10 x i8] c"BREAKING\0A\00"
@.str23 = private unnamed_addr constant [6 x i8] c"i is \00"
@.str24 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str25 = private unnamed_addr constant [12 x i8] c"example.txt\00"
@.str26 = private unnamed_addr constant [14 x i8] c"Hello Orcat!\0A\00"
@.str27 = private unnamed_addr constant [15 x i8] c"File written.\0A\00"
@.str28 = private unnamed_addr constant [8 x i8] c"Hello, \00"
@.str29 = private unnamed_addr constant [3 x i8] c"!\0A\00"
@.str30 = private unnamed_addr constant [6 x i8] c"Alice\00"
@.str31 = private unnamed_addr constant [20 x i8] c"what is your age? \0A\00"
@.str32 = private unnamed_addr constant [8 x i8] c"you're \00"
@.str33 = private unnamed_addr constant [13 x i8] c" years old.\0A\00"
@.str34 = private unnamed_addr constant [9 x i8] c"x += 4: \00"
@.str35 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str36 = private unnamed_addr constant [9 x i8] c"x -= 2: \00"
@.str37 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str38 = private unnamed_addr constant [9 x i8] c"x *= 3: \00"
@.str39 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str40 = private unnamed_addr constant [9 x i8] c"x /= 2: \00"
@.str41 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str42 = private unnamed_addr constant [9 x i8] c"x %= 5: \00"
@.str43 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str44 = private unnamed_addr constant [9 x i8] c"x &= 3: \00"
@.str45 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str46 = private unnamed_addr constant [9 x i8] c"x |= 8: \00"
@.str47 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str48 = private unnamed_addr constant [9 x i8] c"x ^= 1: \00"
@.str49 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str50 = private unnamed_addr constant [10 x i8] c"x <<= 2: \00"
@.str51 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str52 = private unnamed_addr constant [10 x i8] c"x >>= 1: \00"
@.str53 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str54 = private unnamed_addr constant [14 x i8] c"(x & 3) | 8: \00"
@.str55 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str56 = private unnamed_addr constant [9 x i8] c"y >> 1: \00"
@.str57 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str58 = private unnamed_addr constant [37 x i8] c"Assigned values to first 3 pointers.\00"
@.str59 = private unnamed_addr constant [25 x i8] c"Nested allocations done.\00"
@.str60 = private unnamed_addr constant [26 x i8] c"Exited nested autoregion.\00"
@.str61 = private unnamed_addr constant [31 x i8] c"More allocations at top level.\00"
@.str62 = private unnamed_addr constant [29 x i8] c"Exited top-level autoregion.\00"
@.str63 = private unnamed_addr constant [29 x i8] c"Starting crumb autofree test\00"
@.str64 = private unnamed_addr constant [14 x i8] c"Manual forget\00"
@.str65 = private unnamed_addr constant [16 x i8] c"hello ORCatLang\00"
@.str66 = private unnamed_addr constant [11 x i8] c"a is Int: \00"
@.str67 = private unnamed_addr constant [13 x i8] c"b is Float: \00"
@.str68 = private unnamed_addr constant [12 x i8] c"c is Bool: \00"
@.str69 = private unnamed_addr constant [14 x i8] c"d is String: \00"
@.str70 = private unnamed_addr constant [5 x i8] c"int8\00"
@.str71 = private unnamed_addr constant [2 x i8] c"0\00"
@.str72 = private unnamed_addr constant [23 x i8] c"=== TEST: nottest ===\0A\00"
@.str73 = private unnamed_addr constant [20 x i8] c"=== TEST: args ===\0A\00"
@.str74 = private unnamed_addr constant [28 x i8] c"=== TEST: arrayExample ===\0A\00"
@.str75 = private unnamed_addr constant [35 x i8] c"=== TEST: cashregisterexample ===\0A\00"
@.str76 = private unnamed_addr constant [36 x i8] c"=== TEST: continueandbreaktest ===\0A\00"
@.str77 = private unnamed_addr constant [24 x i8] c"=== TEST: filemain ===\0A\00"
@.str78 = private unnamed_addr constant [38 x i8] c"=== TEST: generictypesandstructs ===\0A\00"
@.str79 = private unnamed_addr constant [28 x i8] c"=== TEST: getosandbits ===\0A\00"
@.str80 = private unnamed_addr constant [28 x i8] c"=== TEST: inputexample ===\0A\00"
@.str81 = private unnamed_addr constant [26 x i8] c"=== TEST: mathoptest ===\0A\00"
@.str82 = private unnamed_addr constant [23 x i8] c"=== TEST: memtest ===\0A\00"
@.str83 = private unnamed_addr constant [34 x i8] c"=== TEST: C_RuntimeUnionTest ===\0A\00"
@.str84 = private unnamed_addr constant [21 x i8] c"=== TEST: async ===\0A\00"
@.str85 = private unnamed_addr constant [20 x i8] c"=== TEST: cast ===\0A\00"
@.str86 = private unnamed_addr constant [21 x i8] c"=== TEST: enum1 ===\0A\00"
@.str87 = private unnamed_addr constant [21 x i8] c"=== TEST: enum2 ===\0A\00"
@.str88 = private unnamed_addr constant [28 x i8] c"=== TEST: generic_enum ===\0A\00"
@.str89 = private unnamed_addr constant [27 x i8] c"=== TEST: get_address ===\0A\00"
@.str90 = private unnamed_addr constant [28 x i8] c"=== TEST: multigeneric ===\0A\00"
@.str91 = private unnamed_addr constant [28 x i8] c"=== TEST: tostringtest ===\0A\00"
@.str92 = private unnamed_addr constant [4 x i8] c"int\00"
@.str93 = private unnamed_addr constant [4 x i8] c"int\00"
@.str94 = private unnamed_addr constant [3 x i8] c"64\00"
@.str95 = private unnamed_addr constant [3 x i8] c"32\00"
@.str96 = private unnamed_addr constant [6 x i8] c"float\00"
@.str97 = private unnamed_addr constant [5 x i8] c"bool\00"
@.str98 = private unnamed_addr constant [7 x i8] c"string\00"
@.str99 = private unnamed_addr constant [6 x i8] c"ERROR\00"
@.str100 = private unnamed_addr constant [28 x i8] c"=== ALL TESTS FINISHED ===\0A\00"

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %argc64 = sext i32 %argc to i64
  store i64 %argc64, i64* @orcat_argc_global
  store i8** %argv, i8*** @orcat_argv_global
  call void @orcc_init_runtime()
  %ret64 = call i64 @user_main()
  %ret32 = trunc i64 %ret64 to i32
  ret i32 %ret32
}

	@.oob_msg = private unnamed_addr constant [52 x i8] c"[ORCatCompiler-RT-CHCK]: Index out of bounds error.\00"
	@.null_msg = private unnamed_addr constant [45 x i8] c"[ORCatCompiler-RT-CHCK]: Null pointer deref.\00"
	@.heap_msg = private unnamed_addr constant [67 x i8] c"[ORCatCompiler-RT-HEAP]: Invalid free or heap corruption detected.\00"
	@.alloc_magic = global i64 0
	define void @orcc_oob_abort() {
	entry:
	%tmp_puts = call i32 @puts(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.oob_msg, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
	}
	define void @orcc_null_abort() {
	entry:
	%tmp_puts1 = call i32 @puts(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.null_msg, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
	}
	define void @orcc_init_runtime() {
	entry:
	%t = call i64 @time(i64* null)
	%t32 = trunc i64 %t to i32
	call void @srand(i32 %t32)
	%r = call i32 @rand()
	%r64 = zext i32 %r to i64
	%xor_magic = xor i64 %r64, 16045690984833335023
	store i64 %xor_magic, i64* @.alloc_magic
	ret void
	}
	define i8* @orcc_malloc(i64 %usize) {
	entry:
	%hdr_sz = add i64 %usize, 24
	%ovf = icmp ult i64 %hdr_sz, %usize
	br i1 %ovf, label %oom, label %try_malloc
	oom:
	%tmp_puts_oom = call i32 @puts(i8* getelementptr inbounds ([67 x i8], [67 x i8]* @.heap_msg, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
	try_malloc:
	%raw = call i8* @malloc(i64 %hdr_sz)
	%isnull = icmp eq i8* %raw, null
	br i1 %isnull, label %oom_malloc, label %ok_alloc
	oom_malloc:
	%tmp_puts_oom2 = call i32 @puts(i8* getelementptr inbounds ([67 x i8], [67 x i8]* @.heap_msg, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
	ok_alloc:
	%hdr_ptr = bitcast i8* %raw to i64*
	%global_magic = load i64, i64* @.alloc_magic
	store i64 %global_magic, i64* %hdr_ptr
	%size_slot = getelementptr i8, i8* %raw, i64 8
	%size_slot_i64 = bitcast i8* %size_slot to i64*
	store i64 %usize, i64* %size_slot_i64
	%user_ptr = getelementptr i8, i8* %raw, i64 16
	%footer_ptr = getelementptr i8, i8* %user_ptr, i64 %usize
	%footer_ptr_i64 = bitcast i8* %footer_ptr to i64*
	store i64 %global_magic, i64* %footer_ptr_i64
	ret i8* %user_ptr
	}
	define void @orcc_free(i8* %userptr) {
	entry:
	%is_null = icmp eq i8* %userptr, null
	br i1 %is_null, label %ret_void, label %check_hdr
	check_hdr:
	%raw_hdr = getelementptr i8, i8* %userptr, i64 -16
	%hdr_i64 = bitcast i8* %raw_hdr to i64*
	%magic = load i64, i64* %hdr_i64
	%global_magic_cmp = load i64, i64* @.alloc_magic
	%ok = icmp eq i64 %magic, %global_magic_cmp
	br i1 %ok, label %free_ok, label %free_fail
	free_fail:
	%tmp_puts2 = call i32 @puts(i8* getelementptr inbounds ([67 x i8], [67 x i8]* @.heap_msg, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
	free_ok:
	%size_slot = getelementptr i8, i8* %raw_hdr, i64 8
	%size_i64 = bitcast i8* %size_slot to i64*
	%sz = load i64, i64* %size_i64
	%footer_loc = getelementptr i8, i8* %userptr, i64 %sz
	%footer_i64 = bitcast i8* %footer_loc to i64*
	%footer_val = load i64, i64* %footer_i64
	%ok2 = icmp eq i64 %footer_val, %global_magic_cmp
	br i1 %ok2, label %free_ok2, label %free_fail2
	free_fail2:
	%tmp_puts3 = call i32 @puts(i8* getelementptr inbounds ([67 x i8], [67 x i8]* @.heap_msg, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
	free_ok2:
	store i64 0, i64* %hdr_i64
	%rawptr = bitcast i8* %raw_hdr to i8*
	call void @free(i8* %rawptr)
	ret void
	ret_void:
	ret void
	}
	@.vvolatile_msg = private unnamed_addr constant [69 x i8] c"[ORCatCompiler-RT-CHCK]: Volatile write attempted in vasync (panic).\00"
	define void @orcc_vvolatile_abort() {
	entry:
	%tmp_puts_vv = call i32 @puts(i8* getelementptr inbounds ([69 x i8], [69 x i8]* @.vvolatile_msg, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
	}
	define i64 @orcc_alloc_size(i8* %userptr) {
	entry:
	  %is_null = icmp eq i8* %userptr, null
	  br i1 %is_null, label %ret_zero, label %cont
	cont:
	  %raw_hdr = getelementptr i8, i8* %userptr, i64 -16
	  %hdr_i64 = bitcast i8* %raw_hdr to i64*
	  %magic = load i64, i64* %hdr_i64
	  %global_magic_cmp = load i64, i64* @.alloc_magic
	  %ok = icmp eq i64 %magic, %global_magic_cmp
	  br i1 %ok, label %ok2, label %ret_zero
	ok2:
	  %size_slot = getelementptr i8, i8* %raw_hdr, i64 8
	  %size_i64 = bitcast i8* %size_slot to i64*
	  %sz = load i64, i64* %size_i64
	  ret i64 %sz
	ret_zero:
	  ret i64 0
	}
	%orcc_node = type { i8*, i8*, %orcc_node* }
	@orcc_buckets = global [1024 x %orcc_node*] zeroinitializer
	define void @orcc_register_async(i8* %resume, i8* %handle) {
	entry:
	%szptr = getelementptr %orcc_node, %orcc_node* null, i32 1
	%sz = ptrtoint %orcc_node* %szptr to i64
	%raw = call i8* @malloc(i64 %sz)
	%node = bitcast i8* %raw to %orcc_node*
	%rptr = getelementptr %orcc_node, %orcc_node* %node, i32 0, i32 0
	%hptr = getelementptr %orcc_node, %orcc_node* %node, i32 0, i32 1
	%nptr = getelementptr %orcc_node, %orcc_node* %node, i32 0, i32 2
	store i8* %resume, i8** %rptr
	store i8* %handle, i8** %hptr
	%h_addr = ptrtoint i8* %handle to i64
	%bucket_idx64 = and i64 %h_addr, 1023
	%bucket_idx = trunc i64 %bucket_idx64 to i32
	%slot = getelementptr [1024 x %orcc_node*], [1024 x %orcc_node*]* @orcc_buckets, i32 0, i32 %bucket_idx
	br label %insert_loop
	insert_loop:
	%old_head = load atomic %orcc_node*, %orcc_node** %slot seq_cst, align 8
	store %orcc_node* %old_head, %orcc_node** %nptr
	%pair = cmpxchg %orcc_node** %slot, %orcc_node* %old_head, %orcc_node* %node seq_cst seq_cst
	%succ = extractvalue { %orcc_node*, i1 } %pair, 1
	br i1 %succ, label %insert_done, label %insert_loop
	insert_done:
	ret void
	}
	define void @orcc_remove_and_free_node(%orcc_node* %target, %orcc_node** %slot) {
	entry:
	br label %try_head
	try_head:
	%head = load atomic %orcc_node*, %orcc_node** %slot seq_cst, align 8
	%is_head = icmp eq %orcc_node* %head, %target
	br i1 %is_head, label %remove_head, label %scan_pred
	remove_head:
	%t_nptr = getelementptr %orcc_node, %orcc_node* %target, i32 0, i32 2
	%t_next = load atomic %orcc_node*, %orcc_node** %t_nptr seq_cst, align 8
	%pair = cmpxchg %orcc_node** %slot, %orcc_node* %target, %orcc_node* %t_next seq_cst seq_cst
	%succ = extractvalue { %orcc_node*, i1 } %pair, 1
	br i1 %succ, label %freed, label %try_head
	scan_pred:
	%pred0 = load atomic %orcc_node*, %orcc_node** %slot seq_cst, align 8
	br label %scan_loop
	scan_loop:
	%pred = phi %orcc_node* [ %pred0, %scan_pred ], [ %pred_next, %advance_pred ]
	%pred_is_null = icmp eq %orcc_node* %pred, null
	br i1 %pred_is_null, label %notfound, label %check_pred_next
	check_pred_next:
	%pred_nptr = getelementptr %orcc_node, %orcc_node* %pred, i32 0, i32 2
	%pred_next = load atomic %orcc_node*, %orcc_node** %pred_nptr seq_cst, align 8
	%cmp_pred = icmp eq %orcc_node* %pred_next, %target
	br i1 %cmp_pred, label %try_remove_mid, label %advance_pred
	try_remove_mid:
	%target_nptr = getelementptr %orcc_node, %orcc_node* %target, i32 0, i32 2
	%target_next = load atomic %orcc_node*, %orcc_node** %target_nptr seq_cst, align 8
	%pair2 = cmpxchg %orcc_node** %pred_nptr, %orcc_node* %target, %orcc_node* %target_next seq_cst seq_cst
	%succ2 = extractvalue { %orcc_node*, i1 } %pair2, 1
	br i1 %succ2, label %freed, label %scan_pred
	advance_pred:
	br label %scan_loop
	notfound:
	ret void
	freed:
	%rawptr = bitcast %orcc_node* %target to i8*
	call void @free(i8* %rawptr)
	ret void
	}
	define void @orcc_block_until_complete(i8* %handle) {
	entry:
	%h_addr = ptrtoint i8* %handle to i64
	%bucket_idx64 = and i64 %h_addr, 1023
	%bucket_idx = trunc i64 %bucket_idx64 to i32
	%slot = getelementptr [1024 x %orcc_node*], [1024 x %orcc_node*]* @orcc_buckets, i32 0, i32 %bucket_idx
	br label %scan
	scan:
	%head = load atomic %orcc_node*, %orcc_node** %slot seq_cst, align 8
	br label %scan_loop
	scan_loop:
	%cur = phi %orcc_node* [ %head, %scan ], [ %next, %advance ]
	%isnull = icmp eq %orcc_node* %cur, null
	br i1 %isnull, label %sleep, label %checknode
	checknode:
	%hptr = getelementptr %orcc_node, %orcc_node* %cur, i32 0, i32 1
	%hval = load atomic i8*, i8** %hptr seq_cst, align 8
	%cmp = icmp eq i8* %hval, %handle
	br i1 %cmp, label %invoke, label %advance
	invoke:
	%rptr = getelementptr %orcc_node, %orcc_node* %cur, i32 0, i32 0
	%rval = load atomic i8*, i8** %rptr seq_cst, align 8
	%resume_fn = bitcast i8* %rval to i1 (i8*)*
	%res = call i1 %resume_fn(i8* %handle)
	br i1 %res, label %remove_node, label %scan
	remove_node:
	call void @orcc_remove_and_free_node(%orcc_node* %cur, %orcc_node** %slot)
	br label %done
	advance:
	%nptr2 = getelementptr %orcc_node, %orcc_node* %cur, i32 0, i32 2
	%next = load atomic %orcc_node*, %orcc_node** %nptr2 seq_cst, align 8
	br label %scan_loop
	sleep:
	%tmp_usleep = call i32 @usleep(i32 1000)
	br label %scan
	done:
	ret void
	}
	
