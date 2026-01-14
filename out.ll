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
  %t1 = call i8* @sb_create()
  %t2 = load i8*, i8** %s_addr
  %t3 = call i64 @strlen(i8* %t1)
  %t4 = call i64 @strlen(i8* %t2)
  %t5 = add i64 %t3, %t4
  %t6 = add i64 %t5, 1
  %t7 = call i8* @orcc_malloc(i64 %t6)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t7, i8* %t1, i64 %t3, i1 false)
  %t8 = getelementptr inbounds i8, i8* %t7, i64 %t3
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t8, i8* %t2, i64 %t4, i1 false)
  %t9 = getelementptr inbounds i8, i8* %t7, i64 %t5
  store i8 0, i8* %t9
  ret i8* %t7
}
declare double @tofloat(i64 %x)
declare i64 @toint(double %x)
declare i64 @toint_round(double %x)
declare i8* @rmtrz(double %x)
declare i1 @contains(i8* %var, i8* %detection_char)
declare i64 @countcontain(i8* %x, i8* %detection_char)
define i8* @empty() {
entry:
  %t10 = getelementptr inbounds [1 x i8], [1 x i8]* @.str1, i32 0, i32 0
  %t11 = call i8* @tostr(i8* %t10)
  ret i8* %t11
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
  %t12 = load i8*, i8** %s_addr
  call void @print(i8* %t12)
  call void @newline()
  ret void
}
define void @newline() {
entry:
  %t13 = getelementptr inbounds [2 x i8], [2 x i8]* @.str2, i32 0, i32 0
  call void @print(i8* %t13)
  ret void
}
define i64 @user_main() {
entry:
  %variable_addr = alloca i64
  store i64 0, i64* %variable_addr
  %t14 = add i64 0, 0
  store i64 %t14, i64* %variable_addr
  br label %while_head1
while_head1:
  %t15 = load i64, i64* %variable_addr
  %t16 = add i64 0, 99
  %t17 = icmp ne i64 %t15, %t16
  br i1 %t17, label %while_body2, label %while_end3
while_body2:
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
  %t31 = and i1 %t25, %t30
  br i1 %t31, label %then4, label %else5
then4:
  %t32 = getelementptr inbounds [10 x i8], [10 x i8]* @.str3, i32 0, i32 0
  call void @print(i8* %t32)
  br label %endif6
else5:
  %t33 = load i64, i64* %variable_addr
  %t34 = add i64 0, 3
  %t35 = srem i64 %t33, %t34
  %t36 = add i64 0, 0
  %t37 = icmp eq i64 %t35, %t36
  br i1 %t37, label %then7, label %else8
then7:
  %t38 = getelementptr inbounds [6 x i8], [6 x i8]* @.str4, i32 0, i32 0
  call void @print(i8* %t38)
  br label %endif9
else8:
  %t39 = load i64, i64* %variable_addr
  %t40 = add i64 0, 5
  %t41 = srem i64 %t39, %t40
  %t42 = add i64 0, 0
  %t43 = icmp eq i64 %t41, %t42
  br i1 %t43, label %then10, label %else11
then10:
  %t44 = getelementptr inbounds [6 x i8], [6 x i8]* @.str5, i32 0, i32 0
  call void @print(i8* %t44)
  br label %endif12
else11:
  %t45 = load i64, i64* %variable_addr
  %t46 = call i8* @itostr(i64 %t45)
  call void @print(i8* %t46)
  call void @newline()
  br label %endif12
endif12:
  br label %endif9
endif9:
  br label %endif6
endif6:
  br label %while_head1
while_end3:
  %t47 = add i64 0, 0
  ret i64 %t47
}
@.str_null = private unnamed_addr constant [5 x i8] c"null\00"
@.str1 = private unnamed_addr constant [1 x i8] c"\00"
@.str2 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str3 = private unnamed_addr constant [10 x i8] c"fizzbuzz\0A\00"
@.str4 = private unnamed_addr constant [6 x i8] c"fizz\0A\00"
@.str5 = private unnamed_addr constant [6 x i8] c"buzz\0A\00"

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
	
