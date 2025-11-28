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
	
@orcat_argc_global = global i64 0
@orcat_argv_global = global i8** null


define i64 @user_main() {
entry:
  %t1 = add i64 0, 0
  ret i64 %t1
}
define i32 @main(i32 %argc, i8** %argv) {
entry:
  %argc64 = sext i32 %argc to i64
  store i64 %argc64, i64* @orcat_argc_global
  store i8** %argv, i8*** @orcat_argv_global
  %ret64 = call i64 @user_main()
  %ret32 = trunc i64 %ret64 to i32
  ret i32 %ret32
}
