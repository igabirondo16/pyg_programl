; ModuleID = './kernels/hsto.cl'
source_filename = "./kernels/hsto.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @Histogram_kernel(i32 %0, i32 %1, i32 %2, i32* nocapture readonly %3, i32* nocapture %4, i32* %5) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %7 = tail call i64 @_Z12get_group_idj(i32 0) #3
  %8 = trunc i64 %7 to i32
  %9 = tail call i64 @_Z12get_local_idj(i32 0) #3
  %10 = trunc i64 %9 to i32
  %11 = tail call i64 @_Z14get_local_sizej(i32 0) #3
  %12 = tail call i64 @_Z14get_num_groupsj(i32 0) #3
  %13 = trunc i64 %12 to i32
  %14 = sub nsw i32 %1, %2
  %15 = sdiv i32 %14, %13
  %16 = mul nsw i32 %15, %8
  %17 = add nsw i32 %16, %2
  %18 = add nsw i32 %17, %15
  %19 = icmp sgt i32 %15, %10
  br i1 %19, label %20, label %26

20:                                               ; preds = %6
  %21 = shl i64 %9, 32
  %22 = ashr exact i64 %21, 32
  %23 = shl i64 %11, 32
  %24 = ashr exact i64 %23, 32
  %25 = sext i32 %15 to i64
  br label %34

26:                                               ; preds = %34, %6
  tail call void @_Z7barrierj(i32 1) #4
  %27 = icmp slt i32 %10, %0
  br i1 %27, label %28, label %39

28:                                               ; preds = %26
  %29 = shl i64 %9, 32
  %30 = ashr exact i64 %29, 32
  %31 = shl i64 %11, 32
  %32 = ashr exact i64 %31, 32
  %33 = sext i32 %0 to i64
  br label %48

34:                                               ; preds = %20, %34
  %35 = phi i64 [ %22, %20 ], [ %37, %34 ]
  %36 = getelementptr inbounds i32, i32* %5, i64 %35
  store i32 0, i32* %36, align 4, !tbaa !7
  %37 = add i64 %35, %24
  %38 = icmp slt i64 %37, %25
  br i1 %38, label %34, label %26

39:                                               ; preds = %62, %26
  tail call void @_Z7barrierj(i32 1) #4
  br i1 %19, label %40, label %67

40:                                               ; preds = %39
  %41 = icmp sgt i32 %15, 0
  %42 = sext i32 %15 to i64
  %43 = shl i64 %9, 32
  %44 = ashr exact i64 %43, 32
  %45 = shl i64 %11, 32
  %46 = ashr exact i64 %45, 32
  %47 = sext i32 %17 to i64
  br label %65

48:                                               ; preds = %28, %62
  %49 = phi i64 [ %30, %28 ], [ %63, %62 ]
  %50 = getelementptr inbounds i32, i32* %3, i64 %49
  %51 = load i32, i32* %50, align 4, !tbaa !7
  %52 = mul i32 %51, %1
  %53 = lshr i32 %52, 12
  %54 = icmp uge i32 %53, %17
  %55 = icmp ult i32 %53, %18
  %56 = and i1 %54, %55
  br i1 %56, label %57, label %62

57:                                               ; preds = %48
  %58 = sub i32 %53, %17
  %59 = zext i32 %58 to i64
  %60 = getelementptr inbounds i32, i32* %5, i64 %59
  %61 = tail call i32 @_Z10atomic_addPU7CLlocalVjj(i32* %60, i32 1) #4
  br label %62

62:                                               ; preds = %48, %57
  %63 = add i64 %49, %32
  %64 = icmp slt i64 %63, %33
  br i1 %64, label %48, label %39

65:                                               ; preds = %40, %68
  %66 = phi i64 [ %44, %40 ], [ %74, %68 ]
  br i1 %41, label %76, label %68

67:                                               ; preds = %68, %39
  ret void

68:                                               ; preds = %76, %65
  %69 = phi i32 [ 0, %65 ], [ %78, %76 ]
  %70 = add nsw i64 %66, %47
  %71 = getelementptr inbounds i32, i32* %4, i64 %70
  %72 = load i32, i32* %71, align 4, !tbaa !7
  %73 = add i32 %72, %69
  store i32 %73, i32* %71, align 4, !tbaa !7
  %74 = add i64 %66, %46
  %75 = icmp slt i64 %74, %42
  br i1 %75, label %65, label %67

76:                                               ; preds = %65
  %77 = getelementptr inbounds i32, i32* %5, i64 %66
  %78 = load i32, i32* %77, align 4, !tbaa !7
  br label %68
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
!3 = !{i32 0, i32 0, i32 0, i32 1, i32 1, i32 3}
!4 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!5 = !{!"int", !"int", !"int", !"uint*", !"uint*", !"uint*"}
!6 = !{!"", !"", !"", !"", !"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
