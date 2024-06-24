; ModuleID = './kernels/rscd.cl'
source_filename = "./kernels/rscd.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.flowvector = type { i32, i32, i32, i32 }

; Function Attrs: nofree norecurse nounwind uwtable writeonly
define dso_local i32 @gen_model_param(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, float* nocapture %8) local_unnamed_addr #0 {
  %10 = shl nsw i32 %6, 1
  %11 = sub nsw i32 %2, %10
  %12 = mul nsw i32 %11, %2
  %13 = mul nsw i32 %6, %6
  %14 = mul nsw i32 %3, %3
  %15 = shl nsw i32 %3, 1
  %16 = sub nsw i32 %15, %7
  %17 = mul nsw i32 %16, %7
  %18 = add nuw i32 %13, %14
  %19 = add i32 %18, %12
  %20 = sub i32 %19, %17
  %21 = sitofp i32 %20 to float
  %22 = icmp eq i32 %20, 0
  br i1 %22, label %101, label %23

23:                                               ; preds = %9
  %24 = mul nsw i32 %4, %2
  %25 = mul nsw i32 %7, %1
  %26 = mul i32 %7, %5
  %27 = add i32 %0, %4
  %28 = mul i32 %27, %6
  %29 = add i32 %28, %26
  %30 = add i32 %25, %24
  %31 = sub i32 %30, %29
  %32 = mul nsw i32 %31, %2
  %33 = mul nsw i32 %4, %3
  %34 = mul nsw i32 %6, %1
  %35 = mul nsw i32 %6, %5
  %36 = mul i32 %27, %7
  %37 = sub i32 %33, %34
  %38 = add i32 %37, %35
  %39 = sub i32 %38, %36
  %40 = mul nsw i32 %39, %3
  %41 = mul nsw i32 %7, %7
  %42 = add nuw nsw i32 %41, %13
  %43 = mul nsw i32 %42, %0
  %44 = add i32 %32, %43
  %45 = add i32 %44, %40
  %46 = sitofp i32 %45 to float
  %47 = fdiv float %46, %21, !fpmath !3
  store float %47, float* %8, align 4, !tbaa !4
  %48 = mul nsw i32 %3, %0
  %49 = add i32 %5, %1
  %50 = mul i32 %49, %2
  %51 = sub i32 %48, %33
  %52 = add i32 %51, %34
  %53 = sub i32 %52, %50
  %54 = mul nsw i32 %53, %6
  %55 = mul i32 %2, %0
  %56 = mul i32 %3, %1
  %57 = mul i32 %5, %3
  %58 = add i32 %56, %55
  %59 = add i32 %58, %57
  %60 = sub i32 %30, %59
  %61 = mul nsw i32 %60, %7
  %62 = mul nsw i32 %2, %2
  %63 = add nuw nsw i32 %14, %62
  %64 = mul nsw i32 %63, %5
  %65 = add i32 %61, %64
  %66 = add i32 %65, %54
  %67 = sitofp i32 %66 to float
  %68 = fdiv float %67, %21, !fpmath !3
  %69 = getelementptr inbounds float, float* %8, i64 1
  store float %68, float* %69, align 4, !tbaa !4
  %70 = shl nsw i32 %4, 1
  %71 = sub nsw i32 %0, %70
  %72 = mul nsw i32 %71, %0
  %73 = mul nsw i32 %4, %4
  %74 = shl nsw i32 %5, 1
  %75 = sub nsw i32 %1, %74
  %76 = mul nsw i32 %75, %1
  %77 = mul nsw i32 %5, %5
  %78 = add nuw i32 %77, %73
  %79 = add i32 %78, %72
  %80 = add i32 %79, %76
  %81 = sitofp i32 %80 to float
  %82 = icmp eq i32 %80, 0
  br i1 %82, label %101, label %83

83:                                               ; preds = %23
  %84 = sub nsw i32 %0, %4
  %85 = sub nsw i32 %2, %6
  %86 = mul nsw i32 %85, %84
  %87 = sub nsw i32 %1, %5
  %88 = sub nsw i32 %3, %7
  %89 = mul nsw i32 %88, %87
  %90 = add nsw i32 %89, %86
  %91 = sitofp i32 %90 to float
  %92 = fdiv float %91, %81, !fpmath !3
  %93 = getelementptr inbounds float, float* %8, i64 2
  store float %92, float* %93, align 4, !tbaa !4
  %94 = mul nsw i32 %88, %84
  %95 = sub nsw i32 %5, %1
  %96 = mul nsw i32 %85, %95
  %97 = add nsw i32 %94, %96
  %98 = sitofp i32 %97 to float
  %99 = fdiv float %98, %81, !fpmath !3
  %100 = getelementptr inbounds float, float* %8, i64 3
  store float %99, float* %100, align 4, !tbaa !4
  br label %101

101:                                              ; preds = %23, %9, %83
  %102 = phi i32 [ 1, %83 ], [ 0, %9 ], [ 0, %23 ]
  ret i32 %102
}

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @RANSAC_kernel_block(i32 %0, i32 %1, i32 %2, float %3, i32 %4, float %5, float* nocapture %6, %struct.flowvector* nocapture readonly %7, i32* nocapture readonly %8, i32* nocapture %9, i32* nocapture %10, i32* %11, i32* %12) local_unnamed_addr #1 !kernel_arg_addr_space !8 !kernel_arg_access_qual !9 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !11 {
  %14 = fcmp oge float %5, 0.000000e+00
  %15 = fcmp ole float %5, 1.000000e+00
  %16 = and i1 %14, %15
  %17 = sitofp i32 %4 to float
  %18 = fmul float %17, %5
  %19 = fptosi float %18 to i32
  %20 = select i1 %16, i32 %19, i32 0
  %21 = tail call i64 @_Z12get_local_idj(i32 0) #4
  %22 = trunc i64 %21 to i32
  %23 = tail call i64 @_Z12get_group_idj(i32 0) #4
  %24 = trunc i64 %23 to i32
  %25 = add i32 %20, %24
  %26 = icmp slt i32 %25, %4
  br i1 %26, label %27, label %33

27:                                               ; preds = %13
  %28 = icmp eq i32 %22, 0
  %29 = icmp slt i32 %22, %0
  %30 = sitofp i32 %2 to float
  %31 = sitofp i32 %0 to float
  %32 = fmul float %31, %3
  br label %34

33:                                               ; preds = %142, %13
  ret void

34:                                               ; preds = %27, %142
  %35 = phi i32 [ %25, %27 ], [ %145, %142 ]
  %36 = shl nsw i32 %35, 2
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds float, float* %6, i64 %37
  br i1 %28, label %39, label %73

39:                                               ; preds = %34
  store i32 0, i32* %12, align 4, !tbaa !12
  %40 = shl nsw i32 %35, 1
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds i32, i32* %8, i64 %41
  %43 = load i32, i32* %42, align 4, !tbaa !12
  %44 = sext i32 %43 to i64
  %45 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %44, i32 0
  %46 = load i32, i32* %45, align 4, !tbaa.struct !14
  %47 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %44, i32 1
  %48 = load i32, i32* %47, align 4, !tbaa.struct !14
  %49 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %44, i32 2
  %50 = load i32, i32* %49, align 4, !tbaa.struct !14
  %51 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %44, i32 3
  %52 = load i32, i32* %51, align 4, !tbaa.struct !14
  %53 = or i32 %40, 1
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds i32, i32* %8, i64 %54
  %56 = load i32, i32* %55, align 4, !tbaa !12
  %57 = sext i32 %56 to i64
  %58 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %57, i32 0
  %59 = load i32, i32* %58, align 4, !tbaa.struct !14
  %60 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %57, i32 1
  %61 = load i32, i32* %60, align 4, !tbaa.struct !14
  %62 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %57, i32 2
  %63 = load i32, i32* %62, align 4, !tbaa.struct !14
  %64 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %57, i32 3
  %65 = load i32, i32* %64, align 4, !tbaa.struct !14
  %66 = sub nsw i32 %50, %46
  %67 = sub nsw i32 %52, %48
  %68 = sub nsw i32 %63, %59
  %69 = sub nsw i32 %65, %61
  %70 = tail call i32 @gen_model_param(i32 %46, i32 %48, i32 %66, i32 %67, i32 %59, i32 %61, i32 %68, i32 %69, float* %38)
  %71 = icmp eq i32 %70, 0
  br i1 %71, label %72, label %73

72:                                               ; preds = %39
  store float -2.011000e+03, float* %38, align 4, !tbaa !4
  br label %73

73:                                               ; preds = %39, %72, %34
  tail call void @_Z7barrierj(i32 1) #5
  %74 = load float, float* %38, align 4, !tbaa !4
  %75 = fcmp oeq float %74, -2.011000e+03
  br i1 %75, label %142, label %76

76:                                               ; preds = %73
  br i1 %29, label %77, label %86

77:                                               ; preds = %76
  %78 = getelementptr inbounds float, float* %38, i64 2
  %79 = load float, float* %78, align 4, !tbaa !4
  %80 = getelementptr inbounds float, float* %38, i64 1
  %81 = load float, float* %80, align 4, !tbaa !4
  %82 = getelementptr inbounds float, float* %38, i64 3
  %83 = load float, float* %82, align 4, !tbaa !4
  %84 = tail call i64 @_Z14get_local_sizej(i32 0) #4
  %85 = trunc i64 %84 to i32
  br label %89

86:                                               ; preds = %128, %76
  %87 = phi i32 [ 0, %76 ], [ %129, %128 ]
  %88 = tail call i32 @_Z10atomic_addPU7CLlocalVii(i32* %12, i32 %87) #5
  tail call void @_Z7barrierj(i32 1) #5
  br i1 %28, label %132, label %142

89:                                               ; preds = %77, %128
  %90 = phi i32 [ 0, %77 ], [ %129, %128 ]
  %91 = phi i32 [ %22, %77 ], [ %130, %128 ]
  %92 = sext i32 %91 to i64
  %93 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %92, i32 0
  %94 = load i32, i32* %93, align 4, !tbaa.struct !14
  %95 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %92, i32 1
  %96 = load i32, i32* %95, align 4, !tbaa.struct !14
  %97 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %92, i32 2
  %98 = load i32, i32* %97, align 4, !tbaa.struct !14
  %99 = sitofp i32 %94 to float
  %100 = fsub float %99, %74
  %101 = fmul float %100, %79
  %102 = fptosi float %101 to i32
  %103 = sitofp i32 %96 to float
  %104 = fsub float %103, %81
  %105 = fmul float %104, %83
  %106 = fptosi float %105 to i32
  %107 = sub i32 %94, %98
  %108 = add i32 %107, %102
  %109 = sub i32 %108, %106
  %110 = sitofp i32 %109 to float
  %111 = tail call float @_Z4fabsf(float %110) #4
  %112 = fcmp ult float %111, %30
  br i1 %112, label %113, label %126

113:                                              ; preds = %89
  %114 = fmul float %79, %104
  %115 = fptosi float %114 to i32
  %116 = fmul float %100, %83
  %117 = fptosi float %116 to i32
  %118 = getelementptr inbounds %struct.flowvector, %struct.flowvector* %7, i64 %92, i32 3
  %119 = load i32, i32* %118, align 4, !tbaa.struct !14
  %120 = add i32 %96, %117
  %121 = add i32 %120, %115
  %122 = sub i32 %121, %119
  %123 = sitofp i32 %122 to float
  %124 = tail call float @_Z4fabsf(float %123) #4
  %125 = fcmp ult float %124, %30
  br i1 %125, label %128, label %126

126:                                              ; preds = %113, %89
  %127 = add nsw i32 %90, 1
  br label %128

128:                                              ; preds = %113, %126
  %129 = phi i32 [ %127, %126 ], [ %90, %113 ]
  %130 = add i32 %91, %85
  %131 = icmp slt i32 %130, %0
  br i1 %131, label %89, label %86

132:                                              ; preds = %86
  %133 = load i32, i32* %12, align 4, !tbaa !12
  %134 = sitofp i32 %133 to float
  %135 = fcmp ogt float %32, %134
  br i1 %135, label %136, label %142

136:                                              ; preds = %132
  %137 = tail call i32 @_Z10atomic_addPU8CLglobalVii(i32* %11, i32 1) #5
  %138 = load i32, i32* %12, align 4, !tbaa !12
  %139 = sext i32 %137 to i64
  %140 = getelementptr inbounds i32, i32* %10, i64 %139
  store i32 %138, i32* %140, align 4, !tbaa !12
  %141 = getelementptr inbounds i32, i32* %9, i64 %139
  store i32 %35, i32* %141, align 4, !tbaa !12
  br label %142

142:                                              ; preds = %86, %136, %132, %73
  %143 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
  %144 = trunc i64 %143 to i32
  %145 = add i32 %35, %144
  %146 = icmp slt i32 %145, %4
  br i1 %146, label %34, label %33
}

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_local_idj(i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_group_idj(i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_num_groupsj(i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local void @_Z7barrierj(i32) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
declare dso_local float @_Z4fabsf(float) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_local_sizej(i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local i32 @_Z10atomic_addPU7CLlocalVii(i32*, i32) local_unnamed_addr #3

; Function Attrs: convergent
declare dso_local i32 @_Z10atomic_addPU8CLglobalVii(i32*, i32) local_unnamed_addr #3

attributes #0 = { nofree norecurse nounwind uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { convergent nounwind readnone }
attributes #5 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{!"clang version 10.0.0-4ubuntu1 "}
!3 = !{float 2.500000e+00}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 3}
!9 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!10 = !{!"int", !"int", !"int", !"float", !"int", !"float", !"float*", !"flowvector*", !"int*", !"int*", !"int*", !"int*", !"int*"}
!11 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !6, i64 0}
!14 = !{i64 0, i64 4, !12, i64 4, i64 4, !12, i64 8, i64 4, !12, i64 12, i64 4, !12}
