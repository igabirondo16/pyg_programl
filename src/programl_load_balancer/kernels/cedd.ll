; ModuleID = './kernels/cedd.cl'
source_filename = "./kernels/cedd.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@gaus = dso_local local_unnamed_addr constant [3 x [3 x float]] [[3 x float] [float 6.250000e-02, float 1.250000e-01, float 6.250000e-02], [3 x float] [float 1.250000e-01, float 2.500000e-01, float 1.250000e-01], [3 x float] [float 6.250000e-02, float 1.250000e-01, float 6.250000e-02]], align 16
@sobx = dso_local local_unnamed_addr constant [3 x [3 x i32]] [[3 x i32] [i32 -1, i32 0, i32 1], [3 x i32] [i32 -2, i32 0, i32 2], [3 x i32] [i32 -1, i32 0, i32 1]], align 16
@soby = dso_local local_unnamed_addr constant [3 x [3 x i32]] [[3 x i32] [i32 -1, i32 -2, i32 -1], [3 x i32] zeroinitializer, [3 x i32] [i32 1, i32 2, i32 1]], align 16

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @gaussian_kernel(i8* nocapture readonly %0, i8* nocapture %1, i32 %2, i32 %3, i32* nocapture %4) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %6 = tail call i64 @_Z14get_local_sizej(i32 0) #5
  %7 = trunc i64 %6 to i32
  %8 = tail call i64 @_Z13get_global_idj(i32 1) #5
  %9 = trunc i64 %8 to i32
  %10 = tail call i64 @_Z13get_global_idj(i32 0) #5
  %11 = trunc i64 %10 to i32
  %12 = tail call i64 @_Z12get_local_idj(i32 1) #5
  %13 = trunc i64 %12 to i32
  %14 = add i32 %13, 1
  %15 = tail call i64 @_Z12get_local_idj(i32 0) #5
  %16 = trunc i64 %15 to i32
  %17 = add i32 %16, 1
  %18 = mul nsw i32 %9, %3
  %19 = add nsw i32 %18, %11
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds i8, i8* %0, i64 %20
  %22 = load i8, i8* %21, align 1, !tbaa !7
  %23 = zext i8 %22 to i32
  %24 = add nsw i32 %7, 2
  %25 = mul nsw i32 %14, %24
  %26 = add nsw i32 %25, %17
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds i32, i32* %4, i64 %27
  store i32 %23, i32* %28, align 4, !tbaa !10
  %29 = icmp eq i32 %13, 0
  br i1 %29, label %30, label %57

30:                                               ; preds = %5
  %31 = sub nsw i32 %19, %3
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds i8, i8* %0, i64 %32
  %34 = load i8, i8* %33, align 1, !tbaa !7
  %35 = zext i8 %34 to i32
  %36 = sext i32 %17 to i64
  %37 = getelementptr inbounds i32, i32* %4, i64 %36
  store i32 %35, i32* %37, align 4, !tbaa !10
  %38 = icmp eq i32 %16, 0
  br i1 %38, label %39, label %45

39:                                               ; preds = %30
  %40 = add nsw i32 %31, -1
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds i8, i8* %0, i64 %41
  %43 = load i8, i8* %42, align 1, !tbaa !7
  %44 = zext i8 %43 to i32
  store i32 %44, i32* %4, align 4, !tbaa !10
  br label %92

45:                                               ; preds = %30
  %46 = icmp eq i32 %17, %7
  br i1 %46, label %47, label %94

47:                                               ; preds = %45
  %48 = add nsw i32 %31, 1
  %49 = sext i32 %48 to i64
  %50 = getelementptr inbounds i8, i8* %0, i64 %49
  %51 = load i8, i8* %50, align 1, !tbaa !7
  %52 = zext i8 %51 to i32
  %53 = shl i64 %6, 32
  %54 = add i64 %53, 4294967296
  %55 = ashr exact i64 %54, 32
  %56 = getelementptr inbounds i32, i32* %4, i64 %55
  store i32 %52, i32* %56, align 4, !tbaa !10
  br label %94

57:                                               ; preds = %5
  %58 = icmp eq i32 %14, %7
  br i1 %58, label %59, label %90

59:                                               ; preds = %57
  %60 = add nsw i32 %19, %3
  %61 = sext i32 %60 to i64
  %62 = getelementptr inbounds i8, i8* %0, i64 %61
  %63 = load i8, i8* %62, align 1, !tbaa !7
  %64 = zext i8 %63 to i32
  %65 = add i32 %7, 1
  %66 = mul nsw i32 %65, %24
  %67 = add nsw i32 %17, %66
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds i32, i32* %4, i64 %68
  store i32 %64, i32* %69, align 4, !tbaa !10
  %70 = icmp eq i32 %16, 0
  br i1 %70, label %71, label %79

71:                                               ; preds = %59
  %72 = add nsw i32 %60, -1
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds i8, i8* %0, i64 %73
  %75 = load i8, i8* %74, align 1, !tbaa !7
  %76 = zext i8 %75 to i32
  %77 = sext i32 %66 to i64
  %78 = getelementptr inbounds i32, i32* %4, i64 %77
  store i32 %76, i32* %78, align 4, !tbaa !10
  br label %92

79:                                               ; preds = %59
  %80 = icmp eq i32 %17, %7
  br i1 %80, label %81, label %94

81:                                               ; preds = %79
  %82 = add nsw i32 %60, 1
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds i8, i8* %0, i64 %83
  %85 = load i8, i8* %84, align 1, !tbaa !7
  %86 = zext i8 %85 to i32
  %87 = add i32 %65, %66
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds i32, i32* %4, i64 %88
  store i32 %86, i32* %89, align 4, !tbaa !10
  br label %94

90:                                               ; preds = %57
  %91 = icmp eq i32 %16, 0
  br i1 %91, label %92, label %94

92:                                               ; preds = %39, %71, %90
  %93 = add nsw i32 %19, -1
  br label %100

94:                                               ; preds = %45, %47, %81, %79, %90
  %95 = icmp eq i32 %17, %7
  br i1 %95, label %96, label %109

96:                                               ; preds = %94
  %97 = add nsw i32 %19, 1
  %98 = add i32 %7, 1
  %99 = add i32 %98, %25
  br label %100

100:                                              ; preds = %92, %96
  %101 = phi i32 [ %99, %96 ], [ %25, %92 ]
  %102 = phi i32 [ %97, %96 ], [ %93, %92 ]
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds i8, i8* %0, i64 %103
  %105 = load i8, i8* %104, align 1, !tbaa !7
  %106 = zext i8 %105 to i32
  %107 = sext i32 %101 to i64
  %108 = getelementptr inbounds i32, i32* %4, i64 %107
  store i32 %106, i32* %108, align 4, !tbaa !10
  br label %109

109:                                              ; preds = %100, %94
  tail call void @_Z7barrierj(i32 1) #6
  %110 = mul nsw i32 %24, %13
  %111 = add i32 %110, %16
  %112 = sext i32 %111 to i64
  %113 = getelementptr inbounds i32, i32* %4, i64 %112
  %114 = load i32, i32* %113, align 4, !tbaa !10
  %115 = sitofp i32 %114 to float
  %116 = tail call float @llvm.fmuladd.f32(float %115, float 6.250000e-02, float 0.000000e+00)
  %117 = fptosi float %116 to i32
  %118 = add i32 %111, 1
  %119 = sext i32 %118 to i64
  %120 = getelementptr inbounds i32, i32* %4, i64 %119
  %121 = load i32, i32* %120, align 4, !tbaa !10
  %122 = sitofp i32 %121 to float
  %123 = sitofp i32 %117 to float
  %124 = tail call float @llvm.fmuladd.f32(float %122, float 1.250000e-01, float %123)
  %125 = fptosi float %124 to i32
  %126 = add i32 %111, 2
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds i32, i32* %4, i64 %127
  %129 = load i32, i32* %128, align 4, !tbaa !10
  %130 = sitofp i32 %129 to float
  %131 = sitofp i32 %125 to float
  %132 = tail call float @llvm.fmuladd.f32(float %130, float 6.250000e-02, float %131)
  %133 = fptosi float %132 to i32
  %134 = add i32 %25, %16
  %135 = sext i32 %134 to i64
  %136 = getelementptr inbounds i32, i32* %4, i64 %135
  %137 = load i32, i32* %136, align 4, !tbaa !10
  %138 = sitofp i32 %137 to float
  %139 = sitofp i32 %133 to float
  %140 = tail call float @llvm.fmuladd.f32(float %138, float 1.250000e-01, float %139)
  %141 = fptosi float %140 to i32
  %142 = add i32 %134, 1
  %143 = sext i32 %142 to i64
  %144 = getelementptr inbounds i32, i32* %4, i64 %143
  %145 = load i32, i32* %144, align 4, !tbaa !10
  %146 = sitofp i32 %145 to float
  %147 = sitofp i32 %141 to float
  %148 = tail call float @llvm.fmuladd.f32(float %146, float 2.500000e-01, float %147)
  %149 = fptosi float %148 to i32
  %150 = add i32 %134, 2
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds i32, i32* %4, i64 %151
  %153 = load i32, i32* %152, align 4, !tbaa !10
  %154 = sitofp i32 %153 to float
  %155 = sitofp i32 %149 to float
  %156 = tail call float @llvm.fmuladd.f32(float %154, float 1.250000e-01, float %155)
  %157 = fptosi float %156 to i32
  %158 = add i32 %13, 2
  %159 = mul nsw i32 %158, %24
  %160 = add i32 %159, %16
  %161 = sext i32 %160 to i64
  %162 = getelementptr inbounds i32, i32* %4, i64 %161
  %163 = load i32, i32* %162, align 4, !tbaa !10
  %164 = sitofp i32 %163 to float
  %165 = sitofp i32 %157 to float
  %166 = tail call float @llvm.fmuladd.f32(float %164, float 6.250000e-02, float %165)
  %167 = fptosi float %166 to i32
  %168 = add i32 %160, 1
  %169 = sext i32 %168 to i64
  %170 = getelementptr inbounds i32, i32* %4, i64 %169
  %171 = load i32, i32* %170, align 4, !tbaa !10
  %172 = sitofp i32 %171 to float
  %173 = sitofp i32 %167 to float
  %174 = tail call float @llvm.fmuladd.f32(float %172, float 1.250000e-01, float %173)
  %175 = fptosi float %174 to i32
  %176 = add i32 %160, 2
  %177 = sext i32 %176 to i64
  %178 = getelementptr inbounds i32, i32* %4, i64 %177
  %179 = load i32, i32* %178, align 4, !tbaa !10
  %180 = sitofp i32 %179 to float
  %181 = sitofp i32 %175 to float
  %182 = tail call float @llvm.fmuladd.f32(float %180, float 6.250000e-02, float %181)
  %183 = fptosi float %182 to i32
  %184 = tail call i32 @_Z3maxii(i32 0, i32 %183) #5
  %185 = tail call i32 @_Z3minii(i32 255, i32 %184) #5
  %186 = trunc i32 %185 to i8
  %187 = getelementptr inbounds i8, i8* %1, i64 %20
  store i8 %186, i8* %187, align 1, !tbaa !7
  ret void
}

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_local_sizej(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z13get_global_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_local_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local void @_Z7barrierj(i32) local_unnamed_addr #2

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #3

; Function Attrs: convergent nounwind readnone
declare dso_local i32 @_Z3minii(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i32 @_Z3maxii(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @sobel_kernel(i8* nocapture readonly %0, i8* nocapture %1, i8* nocapture %2, i32 %3, i32 %4, i32* nocapture %5) local_unnamed_addr #0 !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
  %7 = tail call i64 @_Z14get_local_sizej(i32 0) #5
  %8 = trunc i64 %7 to i32
  %9 = tail call i64 @_Z13get_global_idj(i32 1) #5
  %10 = trunc i64 %9 to i32
  %11 = tail call i64 @_Z13get_global_idj(i32 0) #5
  %12 = trunc i64 %11 to i32
  %13 = tail call i64 @_Z12get_local_idj(i32 1) #5
  %14 = trunc i64 %13 to i32
  %15 = add i32 %14, 1
  %16 = tail call i64 @_Z12get_local_idj(i32 0) #5
  %17 = trunc i64 %16 to i32
  %18 = add i32 %17, 1
  %19 = mul nsw i32 %10, %4
  %20 = add nsw i32 %19, %12
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds i8, i8* %0, i64 %21
  %23 = load i8, i8* %22, align 1, !tbaa !7
  %24 = zext i8 %23 to i32
  %25 = add nsw i32 %8, 2
  %26 = mul nsw i32 %15, %25
  %27 = add nsw i32 %26, %18
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds i32, i32* %5, i64 %28
  store i32 %24, i32* %29, align 4, !tbaa !10
  %30 = icmp eq i32 %14, 0
  br i1 %30, label %31, label %58

31:                                               ; preds = %6
  %32 = sub nsw i32 %20, %4
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds i8, i8* %0, i64 %33
  %35 = load i8, i8* %34, align 1, !tbaa !7
  %36 = zext i8 %35 to i32
  %37 = sext i32 %18 to i64
  %38 = getelementptr inbounds i32, i32* %5, i64 %37
  store i32 %36, i32* %38, align 4, !tbaa !10
  %39 = icmp eq i32 %17, 0
  br i1 %39, label %40, label %46

40:                                               ; preds = %31
  %41 = add nsw i32 %32, -1
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds i8, i8* %0, i64 %42
  %44 = load i8, i8* %43, align 1, !tbaa !7
  %45 = zext i8 %44 to i32
  store i32 %45, i32* %5, align 4, !tbaa !10
  br label %93

46:                                               ; preds = %31
  %47 = icmp eq i32 %18, %8
  br i1 %47, label %48, label %95

48:                                               ; preds = %46
  %49 = add nsw i32 %32, 1
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds i8, i8* %0, i64 %50
  %52 = load i8, i8* %51, align 1, !tbaa !7
  %53 = zext i8 %52 to i32
  %54 = shl i64 %7, 32
  %55 = add i64 %54, 4294967296
  %56 = ashr exact i64 %55, 32
  %57 = getelementptr inbounds i32, i32* %5, i64 %56
  store i32 %53, i32* %57, align 4, !tbaa !10
  br label %95

58:                                               ; preds = %6
  %59 = icmp eq i32 %15, %8
  br i1 %59, label %60, label %91

60:                                               ; preds = %58
  %61 = add nsw i32 %20, %4
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds i8, i8* %0, i64 %62
  %64 = load i8, i8* %63, align 1, !tbaa !7
  %65 = zext i8 %64 to i32
  %66 = add nsw i32 %8, 1
  %67 = mul nsw i32 %66, %25
  %68 = add nsw i32 %18, %67
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds i32, i32* %5, i64 %69
  store i32 %65, i32* %70, align 4, !tbaa !10
  %71 = icmp eq i32 %17, 0
  br i1 %71, label %72, label %80

72:                                               ; preds = %60
  %73 = add nsw i32 %61, -1
  %74 = sext i32 %73 to i64
  %75 = getelementptr inbounds i8, i8* %0, i64 %74
  %76 = load i8, i8* %75, align 1, !tbaa !7
  %77 = zext i8 %76 to i32
  %78 = sext i32 %67 to i64
  %79 = getelementptr inbounds i32, i32* %5, i64 %78
  store i32 %77, i32* %79, align 4, !tbaa !10
  br label %93

80:                                               ; preds = %60
  %81 = icmp eq i32 %18, %8
  br i1 %81, label %82, label %95

82:                                               ; preds = %80
  %83 = add nsw i32 %61, 1
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds i8, i8* %0, i64 %84
  %86 = load i8, i8* %85, align 1, !tbaa !7
  %87 = zext i8 %86 to i32
  %88 = add nsw i32 %67, %66
  %89 = sext i32 %88 to i64
  %90 = getelementptr inbounds i32, i32* %5, i64 %89
  store i32 %87, i32* %90, align 4, !tbaa !10
  br label %95

91:                                               ; preds = %58
  %92 = icmp eq i32 %17, 0
  br i1 %92, label %93, label %95

93:                                               ; preds = %40, %72, %91
  %94 = add nsw i32 %20, -1
  br label %101

95:                                               ; preds = %46, %48, %82, %80, %91
  %96 = icmp eq i32 %18, %8
  br i1 %96, label %97, label %110

97:                                               ; preds = %95
  %98 = add nsw i32 %20, 1
  %99 = add nsw i32 %8, 1
  %100 = add nsw i32 %99, %26
  br label %101

101:                                              ; preds = %93, %97
  %102 = phi i32 [ %100, %97 ], [ %26, %93 ]
  %103 = phi i32 [ %98, %97 ], [ %94, %93 ]
  %104 = sext i32 %103 to i64
  %105 = getelementptr inbounds i8, i8* %0, i64 %104
  %106 = load i8, i8* %105, align 1, !tbaa !7
  %107 = zext i8 %106 to i32
  %108 = sext i32 %102 to i64
  %109 = getelementptr inbounds i32, i32* %5, i64 %108
  store i32 %107, i32* %109, align 4, !tbaa !10
  br label %110

110:                                              ; preds = %101, %95
  tail call void @_Z7barrierj(i32 1) #6
  %111 = mul nsw i32 %25, %14
  %112 = add i32 %111, %17
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds i32, i32* %5, i64 %113
  %115 = load i32, i32* %114, align 4, !tbaa !10
  %116 = sub nsw i32 0, %115
  %117 = sitofp i32 %116 to float
  %118 = add i32 %112, 1
  %119 = sext i32 %118 to i64
  %120 = getelementptr inbounds i32, i32* %5, i64 %119
  %121 = load i32, i32* %120, align 4, !tbaa !10
  %122 = mul nsw i32 %121, -2
  %123 = sitofp i32 %122 to float
  %124 = fadd float %117, %123
  %125 = add i32 %112, 2
  %126 = sext i32 %125 to i64
  %127 = getelementptr inbounds i32, i32* %5, i64 %126
  %128 = load i32, i32* %127, align 4, !tbaa !10
  %129 = sitofp i32 %128 to float
  %130 = fadd float %117, %129
  %131 = sub nsw i32 0, %128
  %132 = sitofp i32 %131 to float
  %133 = fadd float %124, %132
  %134 = add i32 %26, %17
  %135 = sext i32 %134 to i64
  %136 = getelementptr inbounds i32, i32* %5, i64 %135
  %137 = load i32, i32* %136, align 4, !tbaa !10
  %138 = mul nsw i32 %137, -2
  %139 = sitofp i32 %138 to float
  %140 = fadd float %130, %139
  %141 = fadd float %133, 0.000000e+00
  %142 = fadd float %140, 0.000000e+00
  %143 = add i32 %134, 2
  %144 = sext i32 %143 to i64
  %145 = getelementptr inbounds i32, i32* %5, i64 %144
  %146 = load i32, i32* %145, align 4, !tbaa !10
  %147 = shl nsw i32 %146, 1
  %148 = sitofp i32 %147 to float
  %149 = fadd float %142, %148
  %150 = add i32 %14, 2
  %151 = mul nsw i32 %150, %25
  %152 = add i32 %151, %17
  %153 = sext i32 %152 to i64
  %154 = getelementptr inbounds i32, i32* %5, i64 %153
  %155 = load i32, i32* %154, align 4, !tbaa !10
  %156 = sub nsw i32 0, %155
  %157 = sitofp i32 %156 to float
  %158 = fadd float %149, %157
  %159 = sitofp i32 %155 to float
  %160 = fadd float %141, %159
  %161 = add i32 %152, 1
  %162 = sext i32 %161 to i64
  %163 = getelementptr inbounds i32, i32* %5, i64 %162
  %164 = load i32, i32* %163, align 4, !tbaa !10
  %165 = fadd float %158, 0.000000e+00
  %166 = shl nsw i32 %164, 1
  %167 = sitofp i32 %166 to float
  %168 = fadd float %160, %167
  %169 = add i32 %152, 2
  %170 = sext i32 %169 to i64
  %171 = getelementptr inbounds i32, i32* %5, i64 %170
  %172 = load i32, i32* %171, align 4, !tbaa !10
  %173 = sitofp i32 %172 to float
  %174 = fadd float %165, %173
  %175 = fadd float %168, %173
  %176 = tail call float @_Z5hypotff(float %174, float %175) #5
  %177 = fptosi float %176 to i32
  %178 = tail call i32 @_Z3maxii(i32 0, i32 %177) #5
  %179 = tail call i32 @_Z3minii(i32 255, i32 %178) #5
  %180 = trunc i32 %179 to i8
  %181 = getelementptr inbounds i8, i8* %1, i64 %21
  store i8 %180, i8* %181, align 1, !tbaa !7
  %182 = tail call float @_Z5atan2ff(float %175, float %174) #5
  %183 = fcmp olt float %182, 0.000000e+00
  br i1 %183, label %184, label %187

184:                                              ; preds = %110
  %185 = fadd float %182, 0x401921FB60000000
  %186 = tail call float @_Z4fmodff(float %185, float 0x401921FB60000000) #5
  br label %187

187:                                              ; preds = %184, %110
  %188 = phi float [ %186, %184 ], [ %182, %110 ]
  %189 = fcmp ugt float %188, 0x3FD921FB60000000
  br i1 %189, label %192, label %190

190:                                              ; preds = %187
  %191 = getelementptr inbounds i8, i8* %2, i64 %21
  store i8 0, i8* %191, align 1, !tbaa !7
  br label %221

192:                                              ; preds = %187
  %193 = fcmp ugt float %188, 0x3FF2D97C80000000
  br i1 %193, label %196, label %194

194:                                              ; preds = %192
  %195 = getelementptr inbounds i8, i8* %2, i64 %21
  store i8 45, i8* %195, align 1, !tbaa !7
  br label %221

196:                                              ; preds = %192
  %197 = fcmp ugt float %188, 0x3FFF6A7A40000000
  br i1 %197, label %200, label %198

198:                                              ; preds = %196
  %199 = getelementptr inbounds i8, i8* %2, i64 %21
  store i8 90, i8* %199, align 1, !tbaa !7
  br label %221

200:                                              ; preds = %196
  %201 = fcmp ugt float %188, 0x4005FDBC00000000
  br i1 %201, label %204, label %202

202:                                              ; preds = %200
  %203 = getelementptr inbounds i8, i8* %2, i64 %21
  store i8 -121, i8* %203, align 1, !tbaa !7
  br label %221

204:                                              ; preds = %200
  %205 = fcmp ugt float %188, 0x400C463AC0000000
  br i1 %205, label %208, label %206

206:                                              ; preds = %204
  %207 = getelementptr inbounds i8, i8* %2, i64 %21
  store i8 0, i8* %207, align 1, !tbaa !7
  br label %221

208:                                              ; preds = %204
  %209 = fcmp ugt float %188, 0x4011475CE0000000
  br i1 %209, label %212, label %210

210:                                              ; preds = %208
  %211 = getelementptr inbounds i8, i8* %2, i64 %21
  store i8 45, i8* %211, align 1, !tbaa !7
  br label %221

212:                                              ; preds = %208
  %213 = fcmp ugt float %188, 0x40146B9C40000000
  br i1 %213, label %216, label %214

214:                                              ; preds = %212
  %215 = getelementptr inbounds i8, i8* %2, i64 %21
  store i8 90, i8* %215, align 1, !tbaa !7
  br label %221

216:                                              ; preds = %212
  %217 = fcmp ugt float %188, 0x40178FDBA0000000
  %218 = getelementptr inbounds i8, i8* %2, i64 %21
  br i1 %217, label %220, label %219

219:                                              ; preds = %216
  store i8 -121, i8* %218, align 1, !tbaa !7
  br label %221

220:                                              ; preds = %216
  store i8 0, i8* %218, align 1, !tbaa !7
  br label %221

221:                                              ; preds = %194, %202, %210, %219, %220, %214, %206, %198, %190
  ret void
}

; Function Attrs: convergent nounwind readnone
declare dso_local float @_Z5hypotff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local float @_Z5atan2ff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local float @_Z4fmodff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @non_max_supp_kernel(i8* nocapture readonly %0, i8* nocapture %1, i8* nocapture readonly %2, i32 %3, i32 %4, i32* nocapture %5) local_unnamed_addr #0 !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
  %7 = tail call i64 @_Z14get_local_sizej(i32 0) #5
  %8 = trunc i64 %7 to i32
  %9 = tail call i64 @_Z13get_global_idj(i32 1) #5
  %10 = trunc i64 %9 to i32
  %11 = tail call i64 @_Z13get_global_idj(i32 0) #5
  %12 = trunc i64 %11 to i32
  %13 = tail call i64 @_Z12get_local_idj(i32 1) #5
  %14 = trunc i64 %13 to i32
  %15 = add i32 %14, 1
  %16 = tail call i64 @_Z12get_local_idj(i32 0) #5
  %17 = trunc i64 %16 to i32
  %18 = add i32 %17, 1
  %19 = mul nsw i32 %10, %4
  %20 = add nsw i32 %19, %12
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds i8, i8* %0, i64 %21
  %23 = load i8, i8* %22, align 1, !tbaa !7
  %24 = zext i8 %23 to i32
  %25 = add nsw i32 %8, 2
  %26 = mul nsw i32 %15, %25
  %27 = add nsw i32 %26, %18
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds i32, i32* %5, i64 %28
  store i32 %24, i32* %29, align 4, !tbaa !10
  %30 = icmp eq i32 %14, 0
  br i1 %30, label %31, label %58

31:                                               ; preds = %6
  %32 = sub nsw i32 %20, %4
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds i8, i8* %0, i64 %33
  %35 = load i8, i8* %34, align 1, !tbaa !7
  %36 = zext i8 %35 to i32
  %37 = sext i32 %18 to i64
  %38 = getelementptr inbounds i32, i32* %5, i64 %37
  store i32 %36, i32* %38, align 4, !tbaa !10
  %39 = icmp eq i32 %17, 0
  br i1 %39, label %40, label %46

40:                                               ; preds = %31
  %41 = add nsw i32 %32, -1
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds i8, i8* %0, i64 %42
  %44 = load i8, i8* %43, align 1, !tbaa !7
  %45 = zext i8 %44 to i32
  store i32 %45, i32* %5, align 4, !tbaa !10
  br label %93

46:                                               ; preds = %31
  %47 = icmp eq i32 %18, %8
  br i1 %47, label %48, label %95

48:                                               ; preds = %46
  %49 = add nsw i32 %32, 1
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds i8, i8* %0, i64 %50
  %52 = load i8, i8* %51, align 1, !tbaa !7
  %53 = zext i8 %52 to i32
  %54 = shl i64 %7, 32
  %55 = add i64 %54, 4294967296
  %56 = ashr exact i64 %55, 32
  %57 = getelementptr inbounds i32, i32* %5, i64 %56
  store i32 %53, i32* %57, align 4, !tbaa !10
  br label %95

58:                                               ; preds = %6
  %59 = icmp eq i32 %15, %8
  br i1 %59, label %60, label %91

60:                                               ; preds = %58
  %61 = add nsw i32 %20, %4
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds i8, i8* %0, i64 %62
  %64 = load i8, i8* %63, align 1, !tbaa !7
  %65 = zext i8 %64 to i32
  %66 = add nsw i32 %8, 1
  %67 = mul nsw i32 %66, %25
  %68 = add nsw i32 %18, %67
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds i32, i32* %5, i64 %69
  store i32 %65, i32* %70, align 4, !tbaa !10
  %71 = icmp eq i32 %17, 0
  br i1 %71, label %72, label %80

72:                                               ; preds = %60
  %73 = add nsw i32 %61, -1
  %74 = sext i32 %73 to i64
  %75 = getelementptr inbounds i8, i8* %0, i64 %74
  %76 = load i8, i8* %75, align 1, !tbaa !7
  %77 = zext i8 %76 to i32
  %78 = sext i32 %67 to i64
  %79 = getelementptr inbounds i32, i32* %5, i64 %78
  store i32 %77, i32* %79, align 4, !tbaa !10
  br label %93

80:                                               ; preds = %60
  %81 = icmp eq i32 %18, %8
  br i1 %81, label %82, label %95

82:                                               ; preds = %80
  %83 = add nsw i32 %61, 1
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds i8, i8* %0, i64 %84
  %86 = load i8, i8* %85, align 1, !tbaa !7
  %87 = zext i8 %86 to i32
  %88 = add nsw i32 %67, %66
  %89 = sext i32 %88 to i64
  %90 = getelementptr inbounds i32, i32* %5, i64 %89
  store i32 %87, i32* %90, align 4, !tbaa !10
  br label %95

91:                                               ; preds = %58
  %92 = icmp eq i32 %17, 0
  br i1 %92, label %93, label %95

93:                                               ; preds = %40, %72, %91
  %94 = add nsw i32 %20, -1
  br label %101

95:                                               ; preds = %46, %48, %82, %80, %91
  %96 = icmp eq i32 %18, %8
  br i1 %96, label %97, label %110

97:                                               ; preds = %95
  %98 = add nsw i32 %20, 1
  %99 = add nsw i32 %8, 1
  %100 = add nsw i32 %99, %26
  br label %101

101:                                              ; preds = %93, %97
  %102 = phi i32 [ %100, %97 ], [ %26, %93 ]
  %103 = phi i32 [ %98, %97 ], [ %94, %93 ]
  %104 = sext i32 %103 to i64
  %105 = getelementptr inbounds i8, i8* %0, i64 %104
  %106 = load i8, i8* %105, align 1, !tbaa !7
  %107 = zext i8 %106 to i32
  %108 = sext i32 %102 to i64
  %109 = getelementptr inbounds i32, i32* %5, i64 %108
  store i32 %107, i32* %109, align 4, !tbaa !10
  br label %110

110:                                              ; preds = %101, %95
  tail call void @_Z7barrierj(i32 1) #6
  %111 = load i32, i32* %29, align 4, !tbaa !10
  %112 = trunc i32 %111 to i8
  %113 = getelementptr inbounds i8, i8* %2, i64 %21
  %114 = load i8, i8* %113, align 1, !tbaa !7
  switch i8 %114, label %182 [
    i8 0, label %115
    i8 45, label %129
    i8 90, label %147
    i8 -121, label %164
  ]

115:                                              ; preds = %110
  %116 = and i32 %111, 255
  %117 = add nsw i32 %27, 1
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds i32, i32* %5, i64 %118
  %120 = load i32, i32* %119, align 4, !tbaa !10
  %121 = icmp sgt i32 %116, %120
  br i1 %121, label %122, label %182

122:                                              ; preds = %115
  %123 = add i32 %26, %17
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds i32, i32* %5, i64 %124
  %126 = load i32, i32* %125, align 4, !tbaa !10
  %127 = icmp sgt i32 %116, %126
  %128 = select i1 %127, i8 %112, i8 0
  br label %182

129:                                              ; preds = %110
  %130 = and i32 %111, 255
  %131 = mul nsw i32 %25, %14
  %132 = add i32 %17, 2
  %133 = add i32 %132, %131
  %134 = sext i32 %133 to i64
  %135 = getelementptr inbounds i32, i32* %5, i64 %134
  %136 = load i32, i32* %135, align 4, !tbaa !10
  %137 = icmp sgt i32 %130, %136
  br i1 %137, label %138, label %182

138:                                              ; preds = %129
  %139 = add i32 %14, 2
  %140 = mul nsw i32 %139, %25
  %141 = add i32 %140, %17
  %142 = sext i32 %141 to i64
  %143 = getelementptr inbounds i32, i32* %5, i64 %142
  %144 = load i32, i32* %143, align 4, !tbaa !10
  %145 = icmp sgt i32 %130, %144
  %146 = select i1 %145, i8 %112, i8 0
  br label %182

147:                                              ; preds = %110
  %148 = and i32 %111, 255
  %149 = mul nsw i32 %25, %14
  %150 = add nsw i32 %18, %149
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds i32, i32* %5, i64 %151
  %153 = load i32, i32* %152, align 4, !tbaa !10
  %154 = icmp sgt i32 %148, %153
  br i1 %154, label %155, label %182

155:                                              ; preds = %147
  %156 = add i32 %14, 2
  %157 = mul nsw i32 %156, %25
  %158 = add nsw i32 %157, %18
  %159 = sext i32 %158 to i64
  %160 = getelementptr inbounds i32, i32* %5, i64 %159
  %161 = load i32, i32* %160, align 4, !tbaa !10
  %162 = icmp sgt i32 %148, %161
  %163 = select i1 %162, i8 %112, i8 0
  br label %182

164:                                              ; preds = %110
  %165 = and i32 %111, 255
  %166 = mul nsw i32 %25, %14
  %167 = add i32 %166, %17
  %168 = sext i32 %167 to i64
  %169 = getelementptr inbounds i32, i32* %5, i64 %168
  %170 = load i32, i32* %169, align 4, !tbaa !10
  %171 = icmp sgt i32 %165, %170
  br i1 %171, label %172, label %182

172:                                              ; preds = %164
  %173 = add i32 %14, 2
  %174 = mul nsw i32 %173, %25
  %175 = add i32 %17, 2
  %176 = add i32 %175, %174
  %177 = sext i32 %176 to i64
  %178 = getelementptr inbounds i32, i32* %5, i64 %177
  %179 = load i32, i32* %178, align 4, !tbaa !10
  %180 = icmp sgt i32 %165, %179
  %181 = select i1 %180, i8 %112, i8 0
  br label %182

182:                                              ; preds = %172, %155, %138, %122, %110, %164, %147, %129, %115
  %183 = phi i8 [ 0, %115 ], [ 0, %129 ], [ 0, %147 ], [ 0, %164 ], [ %112, %110 ], [ %128, %122 ], [ %146, %138 ], [ %163, %155 ], [ %181, %172 ]
  %184 = getelementptr inbounds i8, i8* %1, i64 %21
  store i8 %183, i8* %184, align 1, !tbaa !7
  ret void
}

; Function Attrs: convergent nofree nounwind uwtable
define dso_local spir_kernel void @hyst_kernel(i8* nocapture readonly %0, i8* nocapture %1, i32 %2, i32 %3) local_unnamed_addr #4 !kernel_arg_addr_space !16 !kernel_arg_access_qual !17 !kernel_arg_type !18 !kernel_arg_base_type !18 !kernel_arg_type_qual !19 {
  %5 = tail call i64 @_Z13get_global_idj(i32 1) #5
  %6 = trunc i64 %5 to i32
  %7 = tail call i64 @_Z13get_global_idj(i32 0) #5
  %8 = trunc i64 %7 to i32
  %9 = mul nsw i32 %6, %3
  %10 = add nsw i32 %9, %8
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i8, i8* %0, i64 %11
  %13 = load i8, i8* %12, align 1, !tbaa !7
  %14 = icmp ugt i8 %13, 69
  br i1 %14, label %15, label %17

15:                                               ; preds = %4
  %16 = getelementptr inbounds i8, i8* %1, i64 %11
  store i8 -1, i8* %16, align 1, !tbaa !7
  br label %25

17:                                               ; preds = %4
  %18 = icmp ult i8 %13, 11
  br i1 %18, label %19, label %21

19:                                               ; preds = %17
  %20 = getelementptr inbounds i8, i8* %1, i64 %11
  store i8 0, i8* %20, align 1, !tbaa !7
  br label %25

21:                                               ; preds = %17
  %22 = icmp ugt i8 %13, 39
  %23 = getelementptr inbounds i8, i8* %1, i64 %11
  %24 = sext i1 %22 to i8
  store i8 %24, i8* %23, align 1, !tbaa !7
  br label %25

25:                                               ; preds = %19, %21, %15
  ret void
}

attributes #0 = { convergent nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { convergent nofree nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { convergent nounwind readnone }
attributes #6 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{!"clang version 10.0.0-4ubuntu1 "}
!3 = !{i32 1, i32 1, i32 0, i32 0, i32 3}
!4 = !{!"none", !"none", !"none", !"none", !"none"}
!5 = !{!"uchar*", !"uchar*", !"int", !"int", !"int*"}
!6 = !{!"", !"", !"", !"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{i32 1, i32 1, i32 1, i32 0, i32 0, i32 3}
!13 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!14 = !{!"uchar*", !"uchar*", !"uchar*", !"int", !"int", !"int*"}
!15 = !{!"", !"", !"", !"", !"", !""}
!16 = !{i32 1, i32 1, i32 0, i32 0}
!17 = !{!"none", !"none", !"none", !"none"}
!18 = !{!"uchar*", !"uchar*", !"int", !"int"}
!19 = !{!"", !"", !"", !""}
