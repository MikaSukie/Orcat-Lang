; ModuleID = 'orcat'
source_filename = "main.orcat"

@.str0 = private unnamed_addr constant [3 x i8] c"\n\00"
@.str1 = private unnamed_addr constant [6 x i8] c"hello\00"
declare i32 @printf(i8*)
define void @print(i8* %s) {
entry:
  %s_addr = alloca i8*
  store i8* %s, i8** %s_addr
  %t1 = load i8*, i8** %s_addr
  %t2 = call i32 @printf(i8* %t1)
  ret void
}
define void @println(i8* %text) {
entry:
  %text_addr = alloca i8*
  store i8* %text, i8** %text_addr
  %t3 = load i8*, i8** %text_addr
  call void @print(i8* %t3)
  %t4 = getelementptr inbounds [3 x i8], [3 x i8]* @.str0, i32 0, i32 0
  call void @print(i8* %t4)
  ret void
}
define i32 @main() {
entry:
  %t5 = getelementptr inbounds [6 x i8], [6 x i8]* @.str1, i32 0, i32 0
  call void @println(i8* %t5)
  %t6 = add i32 0, 0
  ret i32 %t6
  ret i32 0
}