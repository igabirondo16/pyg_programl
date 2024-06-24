; ModuleID = './kernels/hsti.cl'
source_filename = "./kernels/hsti.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @Histogram_kernel(i32 %0, i32 %1, i32 %2, float %3, i32* nocapture readonly %4, i32* %5, i32* %6) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %8 = fcmp oge float %3, 0.000000e+00
  %9 = fcmp ole float %3, 1.000000e+00
  %10 = and i1 %8, %9
  %11 = sitofp i32 %2 to float
  %12 = fmul float %11, %3
  %13 = fptosi float %12 to i32
  %14 = select i1 %10, i32 %13, i32 0
  %15 = tail call i64 @_Z12get_local_idj(i32 0) #3
  %16 = trunc i64 %15 to i32
  %17 = tail call i64 @_Z14get_local_sizej(i32 0) #3
  %18 = trunc i64 %17 to i32
  %19 = icmp slt i32 %16, %1
  br i1 %19, label %20, label %26

20:                                               ; preds = %7
  %21 = shl i64 %15, 32
  %22 = ashr exact i64 %21, 32
  %23 = shl i64 %17, 32
  %24 = ashr exact i64 %23, 32
  %25 = sext i32 %1 to i64
  br label %31

26:                                               ; preds = %31, %7
  tail call void @_Z7barrierj(i32 1) #4
  %27 = tail call i64 @_Z12get_group_idj(i32 0) #3
  %28 = trunc i64 %27 to i32
  %29 = add i32 %14, %28
  %30 = icmp slt i32 %29, %2
  br i1 %30, label %43, label %36

31:                                               ; preds = %20, %31
  %32 = phi i64 [ %22, %20 ], [ %34, %31 ]
  %33 = getelementptr inbounds i32, i32* %6, i64 %32
  store i32 0, i32* %33, align 4, !tbaa !7
  %34 = add i64 %32, %24
  %35 = icmp slt i64 %34, %25
  br i1 %35, label %31, label %26

36:                                               ; preds = %43, %26
  tail call void @_Z7barrierj(i32 1) #4
  br i1 %19, label %37, label %59

37:                                               ; preds = %36
  %38 = shl i64 %15, 32
  %39 = ashr exact i64 %38, 32
  %40 = shl i64 %17, 32
  %41 = ashr exact i64 %40, 32
  %42 = sext i32 %1 to i64
  br label %60

43:                                               ; preds = %26, %43
  %44 = phi i32 [ %57, %43 ], [ %29, %26 ]
  %45 = mul nsw i32 %44, %18
  %46 = add nsw i32 %45, %16
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds i32, i32* %4, i64 %47
  %49 = load i32, i32* %48, align 4, !tbaa !7
  %50 = mul i32 %49, %1
  %51 = lshr i32 %50, 12
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds i32, i32* %6, i64 %52
  %54 = tail call i32 @_Z10atomic_addPU7CLlocalVjj(i32* %53, i32 1) #4
  %55 = tail call i64 @_Z14get_num_groupsj(i32 0) #3
  %56 = trunc i64 %55 to i32
  %57 = add i32 %44, %56
  %58 = icmp slt i32 %57, %2
  br i1 %58, label %43, label %36

59:                                               ; preds = %60, %36
  ret void

60:                                               ; preds = %37, %60
  %61 = phi i64 [ %39, %37 ], [ %66, %60 ]
  %62 = getelementptr inbounds i32, i32* %5, i64 %61
  %63 = getelementptr inbounds i32, i32* %6, i64 %61
  %64 = load i32, i32* %63, align 4, !tbaa !7
  %65 = tail call i32 @_Z10atomic_addPU8CLglobalVjj(i32* %62, i32 %64) #4
  %66 = add i64 %61, %41
  %67 = icmp slt i64 %66, %42
  br i1 %67, label %60, label %59
}

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_group_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_local_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_local_sizej(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_num_groupsj(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local void @_Z7barrierj(i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local i32 @_Z10atomic_addPU7CLlocalVjj(i32*, i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local i32 @_Z10atomic_addPU8CLglobalVjj(i32*, i32) local_unnamed_addr #2

attributes #0 = { convergent nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind readnone }
attributes #4 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{!"clang version 10.0.0-4ubuntu1 "}
!3 = !{i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 3}
!4 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!5 = !{!"int", !"int", !"int", !"float", !"uint*", !"uint*", !"uint*"}
!6 = !{!"", !"", !"", !"", !"", !"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
