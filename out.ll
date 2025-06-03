; ModuleID = 'orcat'
source_filename = "main.orcat"

define i1 @and(i1 %a, i1 %b) {
entry:
  %a_addr = alloca i1
  store i1 %a, i1* %a_addr
  %b_addr = alloca i1
  store i1 %b, i1* %b_addr
  %t1 = load i1, i1* %a_addr
  br i1 %t1, label %tern_then1, label %tern_else2
tern_then1:
  %t2 = load i1, i1* %b_addr
  %t3 = add i1 0, %t2
  br label %tern_end3
tern_else2:
  %t4 = add i1 0, 0
  %t5 = add i1 0, %t4
  br label %tern_end3
tern_end3:
  %t6 = phi i1 [%t3, %tern_then1], [%t5, %tern_else2]
  ret i1 %t6
  ret i1 0
}
define i1 @or(i1 %a, i1 %b) {
entry:
  %a_addr = alloca i1
  store i1 %a, i1* %a_addr
  %b_addr = alloca i1
  store i1 %b, i1* %b_addr
  %t7 = load i1, i1* %a_addr
  br i1 %t7, label %tern_then4, label %tern_else5
tern_then4:
  %t8 = add i1 0, 1
  %t9 = add i1 0, %t8
  br label %tern_end6
tern_else5:
  %t10 = load i1, i1* %b_addr
  %t11 = add i1 0, %t10
  br label %tern_end6
tern_end6:
  %t12 = phi i1 [%t9, %tern_then4], [%t11, %tern_else5]
  ret i1 %t12
  ret i1 0
}
declare i8* @sb_create()
declare i8* @sb_append_str(i8*, i8*)
declare i8* @sb_append_int(i8*, i64)
declare i8* @sb_append_float(i8*, double)
declare i8* @sb_append_bool(i8*, i1)
declare i8* @sb_finish(i8*)
declare i8* @itostr(i64)
declare i8* @ftostr(double)
declare i8* @btostr(i1)
declare i8* @tostr(i8*)
declare void @print(i8*)
define void @newline() {
entry:
  %t13 = getelementptr inbounds [2 x i8], [2 x i8]* @.str0, i32 0, i32 0
  call void @print(i8* %t13)
  ret void
}
define i64 @main() {
entry:
  %variable_addr = alloca i64
  %t14 = add i64 0, 0
  store i64 %t14, i64* %variable_addr
  br label %while7
while7:
  %t15 = load i64, i64* %variable_addr
  %t16 = add i64 0, 99
  %t17 = icmp ne i64 %t15, %t16
  br i1 %t17, label %body8, label %endwhile9
body8:
  %t18 = load i64, i64* %variable_addr
  %t19 = add i64 0, 1
  %t20 = add i64 %t18, %t19
  store i64 %t20, i64* %variable_addr
  %t21 = load i64, i64* %variable_addr
  %t22 = add i64 0, 3
  %t23 = srem i64 %t21, %t22
  %t24 = add i64 0, 0
  %t25 = icmp eq i64 %t23, %t24
  %t26 = load i64, i64* %variable_addr
  %t27 = add i64 0, 5
  %t28 = srem i64 %t26, %t27
  %t29 = add i64 0, 0
  %t30 = icmp eq i64 %t28, %t29
  %t31 = call i1 @and(i1 %t25, i1 %t30)
  br i1 %t31, label %then10, label %else11
then10:
  %t32 = getelementptr inbounds [10 x i8], [10 x i8]* @.str1, i32 0, i32 0
  call void @print(i8* %t32)
  br label %endif12
else11:
  %t33 = load i64, i64* %variable_addr
  %t34 = add i64 0, 3
  %t35 = srem i64 %t33, %t34
  %t36 = add i64 0, 0
  %t37 = icmp eq i64 %t35, %t36
  br i1 %t37, label %then13, label %else14
then13:
  %t38 = getelementptr inbounds [6 x i8], [6 x i8]* @.str2, i32 0, i32 0
  call void @print(i8* %t38)
  br label %endif15
else14:
  %t39 = load i64, i64* %variable_addr
  %t40 = add i64 0, 5
  %t41 = srem i64 %t39, %t40
  %t42 = add i64 0, 0
  %t43 = icmp eq i64 %t41, %t42
  br i1 %t43, label %then16, label %else17
then16:
  %t44 = getelementptr inbounds [6 x i8], [6 x i8]* @.str3, i32 0, i32 0
  call void @print(i8* %t44)
  br label %endif18
else17:
  %t45 = load i64, i64* %variable_addr
  %t46 = call i8* @itostr(i64 %t45)
  call void @print(i8* %t46)
  call void @newline()
  br label %endif18
endif18:
  br label %endif15
endif15:
  br label %endif12
endif12:
  br label %while7
endwhile9:
  %t47 = add i64 0, 0
  ret i64 %t47
  ret i64 0
}
@.str0 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str1 = private unnamed_addr constant [10 x i8] c"fizzbuzz\0A\00"
@.str2 = private unnamed_addr constant [6 x i8] c"fizz\0A\00"
@.str3 = private unnamed_addr constant [6 x i8] c"buzz\0A\00"
