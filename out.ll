; ModuleID = 'orcat'
source_filename = "main.orcat"
@__argv_ptr = global i8** null
declare i8* @malloc(i64)
declare void @free(i8*)

declare void @puts(i8*)
declare void @exit(i64)

	@.oob_msg = private unnamed_addr constant [52 x i8] c"[ORCatCompiler-RT-CHCK]: Index out of bounds error.\00"
	define void @orcc_oob_abort() {
	entry:
	  call void @puts(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.oob_msg, i32 0, i32 0))
	  call void @exit(i64 1)
	  unreachable
	}
	
@res = global i8* zeroinitializer
@orcat_argc_global = global i64 0
@orcat_argv_global = global i8** null


declare void @system(i8* %s)
declare i64 @ilength(i64 %integer)
declare i64 @flength(double %decimal)
declare i64 @slength(i8* %str)
declare i1 @streq(i8* %str1, i8* %str2)
declare i64 @orcat_argc()
declare i8* @orcat_argv(i64 %index)
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
declare i8* @sb_create()
declare i8* @sb_append_str(i8* %b, i8* %s)
declare i8* @sb_append_int(i8* %b, i64 %x)
declare i8* @sb_append_float(i8* %b, double %f)
declare i8* @sb_append_bool(i8* %b, i1 %bb)
declare i8* @sb_finish(i8* %b)
declare i8* @itostr(i64 %i)
declare i8* @i8tostr(i8 %i)
declare i8* @i16tostr(i16 %i)
declare i8* @i32tostr(i32 %i)
declare i8* @i64tostr(i64 %i)
declare i8* @ftostr(double %f)
declare i8* @btostr(i1 %b)
declare i8* @tostr(i8* %s)
define i8* @safestring(i8* %s) {
entry:
  %s_addr = alloca i8*
  store i8* %s, i8** %s_addr
  %t1 = call i8* @sb_create()
  %t2 = load i8*, i8** %s_addr
  %t3 = call i8* @sb_append_str(i8* %t1, i8* %t2)
  ret i8* %t3
}
declare double @tofloat(i64 %x)
declare i64 @toint(double %x)
declare i64 @toint_round(double %x)
declare i8* @rmtrz(double %x)
declare i1 @contains(i8* %var, i8* %detection_char)
declare i64 @countcontain(i8* %x, i8* %detection_char)
define i8* @empty() {
entry:
  %t4 = getelementptr inbounds [1 x i8], [1 x i8]* @.str0, i32 0, i32 0
  %t5 = call i8* @tostr(i8* %t4)
  ret i8* %t5
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
  %t6 = load i8*, i8** %s_addr
  call void @print(i8* %t6)
  call void @newline()
  ret void
}
define void @newline() {
entry:
  %t7 = getelementptr inbounds [2 x i8], [2 x i8]* @.str1, i32 0, i32 0
  call void @print(i8* %t7)
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
  %t8 = load i8*, i8** %path_addr
  %t9 = call i1 @file_exists(i8* %t8)
  br i1 %t9, label %then1, label %else2
then1:
  %t10 = load i8*, i8** %path_addr
  %t11 = call i8* @read_file(i8* %t10)
  ret i8* %t11
  br label %endif3
else2:
  %t12 = getelementptr inbounds [1 x i8], [1 x i8]* @.str2, i32 0, i32 0
  ret i8* %t12
  br label %endif3
endif3:
  ret i8* null
}
define i1 @save_text(i8* %path, i8* %text) {
entry:
  %path_addr = alloca i8*
  store i8* %path, i8** %path_addr
  %text_addr = alloca i8*
  store i8* %text, i8** %text_addr
  %t13 = load i8*, i8** %path_addr
  %t14 = load i8*, i8** %text_addr
  %t15 = call i1 @write_file(i8* %t13, i8* %t14)
  ret i1 %t15
}
define i1 @add_text(i8* %path, i8* %text) {
entry:
  %path_addr = alloca i8*
  store i8* %path, i8** %path_addr
  %text_addr = alloca i8*
  store i8* %text, i8** %text_addr
  %t16 = load i8*, i8** %path_addr
  %t17 = load i8*, i8** %text_addr
  %t18 = call i1 @append_file(i8* %t16, i8* %t17)
  ret i1 %t18
}
define i64 @user_main() {
entry:
  %arg1_addr = alloca i8*
  %t19 = add i64 0, 1
  %t20 = call i8* @orcat_argv(i64 %t19)
  store i8* %t20, i8** %arg1_addr
  %t21 = load i8*, i8** %arg1_addr
  %t22 = call i8* @tal(i8* %t21)
  %t23 = getelementptr inbounds [5 x i8], [5 x i8]* @.str3, i32 0, i32 0
  %t24 = call i1 @streq(i8* %t22, i8* %t23)
  %t25 = load i8*, i8** %arg1_addr
  %t26 = call i8* @tal(i8* %t25)
  %t27 = getelementptr inbounds [2 x i8], [2 x i8]* @.str4, i32 0, i32 0
  %t28 = call i1 @streq(i8* %t26, i8* %t27)
  %t29 = or i1 %t24, %t28
  %t30 = load i8*, i8** %arg1_addr
  %t31 = call i8* @tal(i8* %t30)
  %t32 = getelementptr inbounds [3 x i8], [3 x i8]* @.str5, i32 0, i32 0
  %t33 = call i1 @streq(i8* %t31, i8* %t32)
  %t34 = or i1 %t29, %t33
  br i1 %t34, label %then4, label %else5
then4:
  %t35 = getelementptr inbounds [20 x i8], [20 x i8]* @.str6, i32 0, i32 0
  call void @print(i8* %t35)
  call void @newline()
  %t36 = getelementptr inbounds [25 x i8], [25 x i8]* @.str7, i32 0, i32 0
  call void @print(i8* %t36)
  call void @newline()
  br label %endif6
else5:
  %t37 = load i8*, i8** %arg1_addr
  %t38 = call i8* @tal(i8* %t37)
  %t39 = getelementptr inbounds [9 x i8], [9 x i8]* @.str8, i32 0, i32 0
  %t40 = call i1 @streq(i8* %t38, i8* %t39)
  br i1 %t40, label %then7, label %else8
then7:
  %t41 = getelementptr inbounds [1 x i8], [1 x i8]* @.str9, i32 0, i32 0
  store i8* %t41, i8** %arg1_addr
  %t42 = call i8* @empty()
  %t43 = getelementptr inbounds [31 x i8], [31 x i8]* @.str10, i32 0, i32 0
  %t44 = call i8* @sb_append_str(i8* %t42, i8* %t43)
  %t45 = call i8* @get_os()
  %t46 = call i8* @sb_append_str(i8* %t44, i8* %t45)
  %t47 = getelementptr inbounds [5 x i8], [5 x i8]* @.str11, i32 0, i32 0
  %t48 = call i8* @sb_append_str(i8* %t46, i8* %t47)
  call void @print(i8* %t48)
  %t49 = getelementptr inbounds [1 x i8], [1 x i8]* @.str12, i32 0, i32 0
  %t50 = call i8* @input(i8* %t49)
  store i8* %t50, i8** @res
  %t51 = load i8*, i8** @res
  %t52 = call i8* @tal(i8* %t51)
  %t53 = getelementptr inbounds [2 x i8], [2 x i8]* @.str13, i32 0, i32 0
  %t54 = call i1 @streq(i8* %t52, i8* %t53)
  br i1 %t54, label %then10, label %else11
then10:
  %t55 = getelementptr inbounds [29 x i8], [29 x i8]* @.str14, i32 0, i32 0
  call void @print(i8* %t55)
  call void @newline()
  %t56 = add i64 0, 0
  call void @exit(i64 %t56)
  br label %endif12
else11:
  %t57 = getelementptr inbounds [11 x i8], [11 x i8]* @.str15, i32 0, i32 0
  call void @print(i8* %t57)
  call void @newline()
  %t58 = call i8* @get_os()
  %t59 = getelementptr inbounds [6 x i8], [6 x i8]* @.str16, i32 0, i32 0
  %t60 = call i1 @streq(i8* %t58, i8* %t59)
  br i1 %t60, label %then13, label %endif14
then13:
  call void @linpreinstall()
  br label %endif14
endif14:
  br label %endif12
endif12:
  br label %endif9
else8:
  %t61 = load i8*, i8** %arg1_addr
  %t62 = call i8* @tal(i8* %t61)
  %t63 = getelementptr inbounds [7 x i8], [7 x i8]* @.str17, i32 0, i32 0
  %t64 = call i1 @streq(i8* %t62, i8* %t63)
  br i1 %t64, label %then15, label %else16
then15:
  call void @self_update()
  br label %endif17
else16:
  %t65 = load i8*, i8** %arg1_addr
  %t66 = call i8* @tal(i8* %t65)
  %t67 = getelementptr inbounds [5 x i8], [5 x i8]* @.str18, i32 0, i32 0
  %t68 = call i1 @streq(i8* %t66, i8* %t67)
  br i1 %t68, label %then18, label %else19
then18:
  %t69 = call i8* @empty()
  %t70 = getelementptr inbounds [51 x i8], [51 x i8]* @.str19, i32 0, i32 0
  %t71 = call i8* @sb_append_str(i8* %t69, i8* %t70)
  call void @print(i8* %t71)
  call void @newline()
  br label %endif20
else19:
  %t72 = call i8* @empty()
  %t73 = getelementptr inbounds [17 x i8], [17 x i8]* @.str20, i32 0, i32 0
  %t74 = call i8* @sb_append_str(i8* %t72, i8* %t73)
  %t75 = load i8*, i8** %arg1_addr
  %t76 = call i8* @tostr(i8* %t75)
  %t77 = call i8* @sb_append_str(i8* %t74, i8* %t76)
  %t78 = getelementptr inbounds [38 x i8], [38 x i8]* @.str21, i32 0, i32 0
  %t79 = call i8* @sb_append_str(i8* %t77, i8* %t78)
  call void @print(i8* %t79)
  call void @newline()
  br label %endif20
endif20:
  br label %endif17
endif17:
  br label %endif9
endif9:
  br label %endif6
endif6:
  %t80 = add i64 0, 0
  ret i64 %t80
}
define void @linpreinstall() {
entry:
  %t81 = getelementptr inbounds [51 x i8], [51 x i8]* @.str22, i32 0, i32 0
  call void @print(i8* %t81)
  call void @newline()
  %t82 = call i8* @empty()
  %t83 = getelementptr inbounds [36 x i8], [36 x i8]* @.str23, i32 0, i32 0
  %t84 = call i8* @sb_append_str(i8* %t82, i8* %t83)
  %t85 = call i8* @input(i8* %t84)
  store i8* %t85, i8** @res
  call void @newline()
  %t86 = load i8*, i8** @res
  %t87 = call i8* @tal(i8* %t86)
  %t88 = getelementptr inbounds [2 x i8], [2 x i8]* @.str24, i32 0, i32 0
  %t89 = call i1 @streq(i8* %t87, i8* %t88)
  %t90 = load i8*, i8** @res
  %t91 = call i8* @tal(i8* %t90)
  %t92 = getelementptr inbounds [4 x i8], [4 x i8]* @.str25, i32 0, i32 0
  %t93 = call i1 @streq(i8* %t91, i8* %t92)
  %t94 = or i1 %t89, %t93
  br i1 %t94, label %then21, label %else22
then21:
  %t95 = call i8* @empty()
  %t96 = getelementptr inbounds [52 x i8], [52 x i8]* @.str26, i32 0, i32 0
  %t97 = call i8* @sb_append_str(i8* %t95, i8* %t96)
  %t98 = call i8* @input(i8* %t97)
  store i8* %t98, i8** @res
  call void @newline()
  %t99 = load i8*, i8** @res
  %t100 = call i8* @tal(i8* %t99)
  %t101 = getelementptr inbounds [2 x i8], [2 x i8]* @.str27, i32 0, i32 0
  %t102 = call i1 @streq(i8* %t100, i8* %t101)
  %t103 = load i8*, i8** @res
  %t104 = call i8* @tal(i8* %t103)
  %t105 = getelementptr inbounds [4 x i8], [4 x i8]* @.str28, i32 0, i32 0
  %t106 = call i1 @streq(i8* %t104, i8* %t105)
  %t107 = or i1 %t102, %t106
  br i1 %t107, label %then24, label %else25
then24:
  %t108 = add i1 0, 1
  call void @lininstall(i1 %t108)
  br label %endif26
else25:
  %t109 = add i1 0, 0
  call void @lininstall(i1 %t109)
  br label %endif26
endif26:
  br label %endif23
else22:
  %t110 = getelementptr inbounds [38 x i8], [38 x i8]* @.str29, i32 0, i32 0
  call void @print(i8* %t110)
  %t111 = add i64 0, 0
  call void @exit(i64 %t111)
  br label %endif23
endif23:
  ret void
}
define void @lininstall(i1 %mkdir) {
entry:
  %mkdir_addr = alloca i1
  store i1 %mkdir, i1* %mkdir_addr
  %t112 = load i1, i1* %mkdir_addr
  %t113 = add i1 0, 0
  %t114 = icmp eq i1 %t112, %t113
  br i1 %t114, label %then27, label %else28
then27:
  %t115 = getelementptr inbounds [37 x i8], [37 x i8]* @.str30, i32 0, i32 0
  call void @print(i8* %t115)
  call void @newline()
  %t116 = getelementptr inbounds [103 x i8], [103 x i8]* @.str31, i32 0, i32 0
  call void @system(i8* %t116)
  call void @newline()
  %t117 = call i8* @empty()
  %t118 = getelementptr inbounds [45 x i8], [45 x i8]* @.str32, i32 0, i32 0
  %t119 = call i8* @sb_append_str(i8* %t117, i8* %t118)
  %t120 = call i8* @input(i8* %t119)
  store i8* %t120, i8** @res
  %t121 = load i8*, i8** @res
  %t122 = call i8* @tal(i8* %t121)
  %t123 = getelementptr inbounds [2 x i8], [2 x i8]* @.str33, i32 0, i32 0
  %t124 = call i1 @streq(i8* %t122, i8* %t123)
  %t125 = load i8*, i8** @res
  %t126 = call i8* @tal(i8* %t125)
  %t127 = getelementptr inbounds [4 x i8], [4 x i8]* @.str34, i32 0, i32 0
  %t128 = call i1 @streq(i8* %t126, i8* %t127)
  %t129 = or i1 %t124, %t128
  br i1 %t129, label %then30, label %else31
then30:
  call void @linlibinstall()
  %t130 = add i1 0, 0
  call void @asklaunchscript(i1 %t130)
  br label %endif32
else31:
  %t131 = add i1 0, 0
  call void @asklaunchscript(i1 %t131)
  br label %endif32
endif32:
  br label %endif29
else28:
  %t132 = getelementptr inbounds [30 x i8], [30 x i8]* @.str35, i32 0, i32 0
  call void @print(i8* %t132)
  call void @newline()
  %t133 = getelementptr inbounds [134 x i8], [134 x i8]* @.str36, i32 0, i32 0
  call void @system(i8* %t133)
  call void @newline()
  %t134 = call i8* @empty()
  %t135 = getelementptr inbounds [45 x i8], [45 x i8]* @.str37, i32 0, i32 0
  %t136 = call i8* @sb_append_str(i8* %t134, i8* %t135)
  %t137 = call i8* @input(i8* %t136)
  store i8* %t137, i8** @res
  %t138 = load i8*, i8** @res
  %t139 = call i8* @tal(i8* %t138)
  %t140 = getelementptr inbounds [2 x i8], [2 x i8]* @.str38, i32 0, i32 0
  %t141 = call i1 @streq(i8* %t139, i8* %t140)
  %t142 = load i8*, i8** @res
  %t143 = call i8* @tal(i8* %t142)
  %t144 = getelementptr inbounds [4 x i8], [4 x i8]* @.str39, i32 0, i32 0
  %t145 = call i1 @streq(i8* %t143, i8* %t144)
  %t146 = or i1 %t141, %t145
  br i1 %t146, label %then33, label %else34
then33:
  %t147 = getelementptr inbounds [711 x i8], [711 x i8]* @.str40, i32 0, i32 0
  call void @system(i8* %t147)
  call void @newline()
  %t148 = add i1 0, 1
  call void @asklaunchscript(i1 %t148)
  br label %endif35
else34:
  %t149 = add i1 0, 1
  call void @asklaunchscript(i1 %t149)
  br label %endif35
endif35:
  br label %endif29
endif29:
  ret void
}
define void @asklaunchscript(i1 %dir) {
entry:
  %dir_addr = alloca i1
  store i1 %dir, i1* %dir_addr
  %t150 = call i8* @empty()
  %t151 = getelementptr inbounds [2 x i8], [2 x i8]* @.str41, i32 0, i32 0
  %t152 = call i8* @sb_append_str(i8* %t150, i8* %t151)
  %t153 = getelementptr inbounds [73 x i8], [73 x i8]* @.str42, i32 0, i32 0
  %t154 = call i8* @sb_append_str(i8* %t152, i8* %t153)
  %t155 = call i8* @input(i8* %t154)
  store i8* %t155, i8** @res
  %t156 = load i8*, i8** @res
  %t157 = call i8* @tal(i8* %t156)
  %t158 = getelementptr inbounds [2 x i8], [2 x i8]* @.str43, i32 0, i32 0
  %t159 = call i1 @streq(i8* %t157, i8* %t158)
  %t160 = load i8*, i8** @res
  %t161 = call i8* @tal(i8* %t160)
  %t162 = getelementptr inbounds [4 x i8], [4 x i8]* @.str44, i32 0, i32 0
  %t163 = call i1 @streq(i8* %t161, i8* %t162)
  %t164 = or i1 %t159, %t163
  br i1 %t164, label %then36, label %else37
then36:
  %t165 = load i1, i1* %dir_addr
  %t166 = add i1 0, 1
  %t167 = icmp eq i1 %t165, %t166
  br i1 %t167, label %then39, label %else40
then39:
  %t168 = add i1 0, 1
  call void @launchtempinstall(i1 %t168)
  br label %endif41
else40:
  %t169 = add i1 0, 0
  call void @launchtempinstall(i1 %t169)
  br label %endif41
endif41:
  br label %endif38
else37:
  call void @thanks()
  %t170 = add i64 0, 0
  call void @exit(i64 %t170)
  br label %endif38
endif38:
  ret void
}
define void @launchtempinstall(i1 %dir) {
entry:
  %dir_addr = alloca i1
  store i1 %dir, i1* %dir_addr
  %t171 = load i1, i1* %dir_addr
  %t172 = add i1 0, 1
  %t173 = icmp eq i1 %t171, %t172
  br i1 %t173, label %then42, label %else43
then42:
  %t174 = call i8* @get_os()
  %t175 = getelementptr inbounds [6 x i8], [6 x i8]* @.str45, i32 0, i32 0
  %t176 = call i1 @streq(i8* %t174, i8* %t175)
  br i1 %t176, label %then45, label %else46
then45:
  %t177 = getelementptr inbounds [205 x i8], [205 x i8]* @.str46, i32 0, i32 0
  call void @system(i8* %t177)
  call void @thanks()
  %t178 = add i64 0, 0
  call void @exit(i64 %t178)
  br label %endif47
else46:
  %t179 = call i8* @get_os()
  %t180 = getelementptr inbounds [8 x i8], [8 x i8]* @.str47, i32 0, i32 0
  %t181 = call i1 @streq(i8* %t179, i8* %t180)
  br i1 %t181, label %then48, label %endif49
then48:
  %t182 = getelementptr inbounds [209 x i8], [209 x i8]* @.str48, i32 0, i32 0
  call void @system(i8* %t182)
  call void @thanks()
  %t183 = add i64 0, 0
  call void @exit(i64 %t183)
  br label %endif49
endif49:
  br label %endif47
endif47:
  br label %endif44
else43:
  %t184 = call i8* @get_os()
  %t185 = getelementptr inbounds [6 x i8], [6 x i8]* @.str49, i32 0, i32 0
  %t186 = call i1 @streq(i8* %t184, i8* %t185)
  br i1 %t186, label %then50, label %else51
then50:
  %t187 = getelementptr inbounds [191 x i8], [191 x i8]* @.str50, i32 0, i32 0
  call void @system(i8* %t187)
  call void @thanks()
  %t188 = add i64 0, 0
  call void @exit(i64 %t188)
  br label %endif52
else51:
  %t189 = call i8* @get_os()
  %t190 = getelementptr inbounds [8 x i8], [8 x i8]* @.str51, i32 0, i32 0
  %t191 = call i1 @streq(i8* %t189, i8* %t190)
  br i1 %t191, label %then53, label %endif54
then53:
  %t192 = getelementptr inbounds [195 x i8], [195 x i8]* @.str52, i32 0, i32 0
  call void @system(i8* %t192)
  call void @thanks()
  %t193 = add i64 0, 0
  call void @exit(i64 %t193)
  br label %endif54
endif54:
  br label %endif52
endif52:
  br label %endif44
endif44:
  ret void
}
define void @thanks() {
entry:
  %t194 = getelementptr inbounds [36 x i8], [36 x i8]* @.str53, i32 0, i32 0
  call void @system(i8* %t194)
  %t195 = getelementptr inbounds [49 x i8], [49 x i8]* @.str54, i32 0, i32 0
  call void @print(i8* %t195)
  call void @newline()
  %t196 = getelementptr inbounds [33 x i8], [33 x i8]* @.str55, i32 0, i32 0
  call void @print(i8* %t196)
  call void @newline()
  %t197 = getelementptr inbounds [33 x i8], [33 x i8]* @.str56, i32 0, i32 0
  call void @print(i8* %t197)
  call void @newline()
  %t198 = getelementptr inbounds [33 x i8], [33 x i8]* @.str57, i32 0, i32 0
  call void @print(i8* %t198)
  call void @newline()
  %t199 = getelementptr inbounds [33 x i8], [33 x i8]* @.str58, i32 0, i32 0
  call void @print(i8* %t199)
  call void @newline()
  %t200 = getelementptr inbounds [33 x i8], [33 x i8]* @.str59, i32 0, i32 0
  call void @print(i8* %t200)
  call void @newline()
  %t201 = getelementptr inbounds [33 x i8], [33 x i8]* @.str60, i32 0, i32 0
  call void @print(i8* %t201)
  call void @newline()
  ret void
}
define void @linlibinstall() {
entry:
  %t202 = getelementptr inbounds [41 x i8], [41 x i8]* @.str61, i32 0, i32 0
  call void @print(i8* %t202)
  call void @newline()
  %t203 = getelementptr inbounds [697 x i8], [697 x i8]* @.str62, i32 0, i32 0
  call void @system(i8* %t203)
  ret void
}
define void @self_update() {
entry:
  %t204 = getelementptr inbounds [32 x i8], [32 x i8]* @.str63, i32 0, i32 0
  call void @print(i8* %t204)
  call void @newline()
  %code_addr = alloca i8*
  %t205 = call i8* @sb_create()
  store i8* %t205, i8** %code_addr
  %t206 = load i8*, i8** %code_addr
  %t207 = getelementptr inbounds [11 x i8], [11 x i8]* @.str64, i32 0, i32 0
  %t208 = call i8* @sb_append_str(i8* %t206, i8* %t207)
  store i8* %t208, i8** %code_addr
  %t209 = load i8*, i8** %code_addr
  %t210 = getelementptr inbounds [9 x i8], [9 x i8]* @.str65, i32 0, i32 0
  %t211 = call i8* @sb_append_str(i8* %t209, i8* %t210)
  store i8* %t211, i8** %code_addr
  %t212 = load i8*, i8** %code_addr
  %t213 = getelementptr inbounds [36 x i8], [36 x i8]* @.str66, i32 0, i32 0
  %t214 = call i8* @sb_append_str(i8* %t212, i8* %t213)
  store i8* %t214, i8** %code_addr
  %t215 = load i8*, i8** %code_addr
  %t216 = getelementptr inbounds [98 x i8], [98 x i8]* @.str67, i32 0, i32 0
  %t217 = call i8* @sb_append_str(i8* %t215, i8* %t216)
  store i8* %t217, i8** %code_addr
  %t218 = load i8*, i8** %code_addr
  %t219 = getelementptr inbounds [15 x i8], [15 x i8]* @.str68, i32 0, i32 0
  %t220 = call i8* @sb_append_str(i8* %t218, i8* %t219)
  store i8* %t220, i8** %code_addr
  %t221 = load i8*, i8** %code_addr
  %t222 = getelementptr inbounds [56 x i8], [56 x i8]* @.str69, i32 0, i32 0
  %t223 = call i8* @sb_append_str(i8* %t221, i8* %t222)
  store i8* %t223, i8** %code_addr
  %script_addr = alloca i8*
  %t224 = load i8*, i8** %code_addr
  %t225 = call i8* @sb_finish(i8* %t224)
  store i8* %t225, i8** %script_addr
  %t226 = getelementptr inbounds [16 x i8], [16 x i8]* @.str70, i32 0, i32 0
  %t227 = load i8*, i8** %script_addr
  %t228 = call i1 @write_file(i8* %t226, i8* %t227)
  %t229 = getelementptr inbounds [16 x i8], [16 x i8]* @.str71, i32 0, i32 0
  %t230 = load i8*, i8** %script_addr
  %t231 = call i1 @write_file(i8* %t229, i8* %t230)
  %t232 = xor i1 %t231, true
  br i1 %t232, label %then55, label %endif56
then55:
  %t233 = getelementptr inbounds [32 x i8], [32 x i8]* @.str72, i32 0, i32 0
  call void @print(i8* %t233)
  call void @newline()
  %t234 = load i8*, i8** %script_addr
  call void @free_str(i8* %t234)
  ret void
  br label %endif56
endif56:
  %t235 = load i8*, i8** %script_addr
  call void @free_str(i8* %t235)
  %t236 = getelementptr inbounds [48 x i8], [48 x i8]* @.str73, i32 0, i32 0
  call void @system(i8* %t236)
  %t237 = getelementptr inbounds [54 x i8], [54 x i8]* @.str74, i32 0, i32 0
  call void @print(i8* %t237)
  call void @newline()
  %t238 = add i64 0, 0
  call void @exit(i64 %t238)
  ret void
}
@.str0 = private unnamed_addr constant [1 x i8] c"\00"
@.str1 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str2 = private unnamed_addr constant [1 x i8] c"\00"
@.str3 = private unnamed_addr constant [5 x i8] c"help\00"
@.str4 = private unnamed_addr constant [2 x i8] c"h\00"
@.str5 = private unnamed_addr constant [3 x i8] c"-h\00"
@.str6 = private unnamed_addr constant [20 x i8] c"available commands:\00"
@.str7 = private unnamed_addr constant [25 x i8] c"> ossemble, help, update\00"
@.str8 = private unnamed_addr constant [9 x i8] c"ossemble\00"
@.str9 = private unnamed_addr constant [1 x i8] c"\00"
@.str10 = private unnamed_addr constant [31 x i8] c"OS override? system detected: \00"
@.str11 = private unnamed_addr constant [5 x i8] c">>: \00"
@.str12 = private unnamed_addr constant [1 x i8] c"\00"
@.str13 = private unnamed_addr constant [2 x i8] c"y\00"
@.str14 = private unnamed_addr constant [29 x i8] c"override not yet implemented\00"
@.str15 = private unnamed_addr constant [11 x i8] c"continuing\00"
@.str16 = private unnamed_addr constant [6 x i8] c"linux\00"
@.str17 = private unnamed_addr constant [7 x i8] c"update\00"
@.str18 = private unnamed_addr constant [5 x i8] c"null\00"
@.str19 = private unnamed_addr constant [51 x i8] c"please type an argument. EX: -h, ossemble, update.\00"
@.str20 = private unnamed_addr constant [17 x i8] c"Your argument: \22\00"
@.str21 = private unnamed_addr constant [38 x i8] c"\22 either does not exist or is a typo.\00"
@.str22 = private unnamed_addr constant [51 x i8] c"Please make sure wget is installed on your system.\00"
@.str23 = private unnamed_addr constant [36 x i8] c"Are you in your desired directory? \00"
@.str24 = private unnamed_addr constant [2 x i8] c"y\00"
@.str25 = private unnamed_addr constant [4 x i8] c"yes\00"
@.str26 = private unnamed_addr constant [52 x i8] c"make a sub env directory? (will be named ocatenv): \00"
@.str27 = private unnamed_addr constant [2 x i8] c"y\00"
@.str28 = private unnamed_addr constant [4 x i8] c"yes\00"
@.str29 = private unnamed_addr constant [38 x i8] c"quitting... since not in desired dir.\00"
@.str30 = private unnamed_addr constant [37 x i8] c"Installing into current directory...\00"
@.str31 = private unnamed_addr constant [103 x i8] c"wget -O ORCC.bin https://github.com/MikaLorielle/Orcat-Lang/releases/download/ORCC-mainstream/ORCC.bin\00"
@.str32 = private unnamed_addr constant [45 x i8] c"Do you want default libs with the compiler? \00"
@.str33 = private unnamed_addr constant [2 x i8] c"y\00"
@.str34 = private unnamed_addr constant [4 x i8] c"yes\00"
@.str35 = private unnamed_addr constant [30 x i8] c"Installing into ./ocatenv/...\00"
@.str36 = private unnamed_addr constant [134 x i8] c"mkdir ocatenv && cd ocatenv && wget -O ORCC.bin https://github.com/MikaLorielle/Orcat-Lang/releases/download/ORCC-mainstream/ORCC.bin\00"
@.str37 = private unnamed_addr constant [45 x i8] c"Do you want default libs with the compiler? \00"
@.str38 = private unnamed_addr constant [2 x i8] c"y\00"
@.str39 = private unnamed_addr constant [4 x i8] c"yes\00"
@.str40 = private unnamed_addr constant [711 x i8] c"cd ocatenv && wget -O stdlib.c https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/stdlib.c && wget -O C_io.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_io.sorcat && wget -O C_fileio.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_fileio.sorcat && wget -O C_mem.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_mem.sorcat && wget -O C_types.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_types.sorcat && wget -O std.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/std.sorcat && wget -O C_union.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_union.sorcat\00"
@.str41 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str42 = private unnamed_addr constant [73 x i8] c"Do you want launch script (templates)? (will install according to OS.): \00"
@.str43 = private unnamed_addr constant [2 x i8] c"y\00"
@.str44 = private unnamed_addr constant [4 x i8] c"yes\00"
@.str45 = private unnamed_addr constant [6 x i8] c"linux\00"
@.str46 = private unnamed_addr constant [205 x i8] c"cd ocatenv && wget -O compile.sh https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/compile.sh && wget -O Rcompile.sh https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/Rcompile.sh\00"
@.str47 = private unnamed_addr constant [8 x i8] c"windows\00"
@.str48 = private unnamed_addr constant [209 x i8] c"cd ocatenv && wget -O compile.bat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/compile.bat && wget -O Rcompile.bat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/Rcompile.bat\00"
@.str49 = private unnamed_addr constant [6 x i8] c"linux\00"
@.str50 = private unnamed_addr constant [191 x i8] c"wget -O compile.sh https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/compile.sh && wget -O Rcompile.sh https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/Rcompile.sh\00"
@.str51 = private unnamed_addr constant [8 x i8] c"windows\00"
@.str52 = private unnamed_addr constant [195 x i8] c"wget -O compile.bat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/compile.bat && wget -O Rcompile.bat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/Rcompile.bat\00"
@.str53 = private unnamed_addr constant [36 x i8] c"find . -type f -exec chmod +x {} \\;\00"
@.str54 = private unnamed_addr constant [49 x i8] c"Thanks for downloading ORCat Language! Have fun!\00"
@.str55 = private unnamed_addr constant [33 x i8] c"   ____  _____   _____      _   \00"
@.str56 = private unnamed_addr constant [33 x i8] c"  / __ \\|  __ \\ / ____|    | |  \00"
@.str57 = private unnamed_addr constant [33 x i8] c" | |  | | |__) | |     __ _| |_ \00"
@.str58 = private unnamed_addr constant [33 x i8] c" | |  | |  _  /| |    / _` | __|\00"
@.str59 = private unnamed_addr constant [33 x i8] c" | |__| | | \\ \\| |___| (_| | |_ \00"
@.str60 = private unnamed_addr constant [33 x i8] c"  \\____/|_|  \\_\\\\_____\\__,_|\\__|\00"
@.str61 = private unnamed_addr constant [41 x i8] c"[IMPORTANT] MAKE SURE WGET is installed!\00"
@.str62 = private unnamed_addr constant [697 x i8] c"wget -O stdlib.c https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/stdlib.c && wget -O C_io.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_io.sorcat && wget -O C_fileio.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_fileio.sorcat && wget -O C_mem.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_mem.sorcat && wget -O C_types.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_types.sorcat && wget -O std.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/std.sorcat && wget -O C_union.sorcat https://raw.githubusercontent.com/MikaLorielle/Orcat-Lang/main/C_union.sorcat\00"
@.str63 = private unnamed_addr constant [32 x i8] c"Preparing background updater...\00"
@.str64 = private unnamed_addr constant [11 x i8] c"#!/bin/sh\0A\00"
@.str65 = private unnamed_addr constant [9 x i8] c"sleep 1\0A\00"
@.str66 = private unnamed_addr constant [36 x i8] c"echo '[UPDATER] Replacing ocat...'\0A\00"
@.str67 = private unnamed_addr constant [98 x i8] c"wget -O ocat https://github.com/MikaLorielle/Orcat-Lang/releases/download/ORCC-BETA-NEWGEN/ocatl\0A\00"
@.str68 = private unnamed_addr constant [15 x i8] c"chmod +x ocat\0A\00"
@.str69 = private unnamed_addr constant [56 x i8] c"echo '[UPDATER] Update complete. Press enter to exit.'\0A\00"
@.str70 = private unnamed_addr constant [16 x i8] c"ocat_updater.sh\00"
@.str71 = private unnamed_addr constant [16 x i8] c"ocat_updater.sh\00"
@.str72 = private unnamed_addr constant [32 x i8] c"Failed to write updater script.\00"
@.str73 = private unnamed_addr constant [48 x i8] c"chmod +x ocat_updater.sh && ./ocat_updater.sh &\00"
@.str74 = private unnamed_addr constant [54 x i8] c"Update started in background. Exiting current ocat...\00"

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %argc64 = sext i32 %argc to i64
  store i64 %argc64, i64* @orcat_argc_global
  store i8** %argv, i8*** @orcat_argv_global
  %ret64 = call i64 @user_main()
  %ret32 = trunc i64 %ret64 to i32
  ret i32 %ret32
}
