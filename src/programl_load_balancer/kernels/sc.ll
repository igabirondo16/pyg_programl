; ModuleID = './kernels/sc.cl'
source_filename = "./kernels/sc.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: convergent nounwind uwtable
define dso_local void @reduce(i32* nocapture %0, i32 %1, i32* nocapture %2) local_unnamed_addr #0 {
  %4 = tail call i64 @_Z12get_local_idj(i32 0) #4
  %5 = trunc i64 %4 to i32
  %6 = tail call i64 @_Z14get_local_sizej(i32 0) #4
  %7 = trunc i64 %6 to i32
  %8 = shl i64 %4, 32
  %9 = ashr exact i64 %8, 32
  %10 = getelementptr inbounds i32, i32* %2, i64 %9
  store i32 %1, i32* %10, align 4, !tbaa !3
  tail call void @_Z7barrierj(i32 1) #5
  %11 = icmp sgt i32 %7, 1
  br i1 %11, label %14, label %12

12:                                               ; preds = %25, %3
  %13 = icmp eq i32 %5, 0
  br i1 %13, label %27, label %29

14:                                               ; preds = %3, %25
  %15 = phi i32 [ %16, %25 ], [ %7, %3 ]
  %16 = ashr i32 %15, 1
  %17 = icmp sgt i32 %16, %5
  br i1 %17, label %18, label %25

18:                                               ; preds = %14
  %19 = add nsw i32 %16, %5
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds i32, i32* %2, i64 %20
  %22 = load i32, i32* %21, align 4, !tbaa !3
  %23 = load i32, i32* %10, align 4, !tbaa !3
  %24 = add nsw i32 %23, %22
  store i32 %24, i32* %10, align 4, !tbaa !3
  br label %25

25:                                               ; preds = %18, %14
  tail call void @_Z7barrierj(i32 1) #5
  %26 = icmp sgt i32 %15, 3
  br i1 %26, label %14, label %12

27:                                               ; preds = %12
  %28 = load i32, i32* %2, align 4, !tbaa !3
  store i32 %28, i32* %0, align 4, !tbaa !3
  br label %29

29:                                               ; preds = %27, %12
  ret void
}

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_local_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_local_sizej(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local void @_Z7barrierj(i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind uwtable
define dso_local i32 @block_binary_prefix_sums(i32* nocapture %0, i32 %1, i32* nocapture %2) local_unnamed_addr #0 {
  %4 = tail call i64 @_Z12get_local_idj(i32 0) #4
  %5 = getelementptr inbounds i32, i32* %2, i64 %4
  store i32 %1, i32* %5, align 4, !tbaa !3
  %6 = tail call i64 @_Z14get_local_sizej(i32 0) #4
  %7 = trunc i64 %6 to i32
  %8 = icmp sgt i32 %7, 1
  br i1 %8, label %9, label %14

9:                                                ; preds = %3
  %10 = trunc i64 %4 to i32
  %11 = shl i32 %10, 1
  %12 = or i32 %11, 1
  %13 = add i32 %11, 2
  br label %24

14:                                               ; preds = %42, %3
  %15 = phi i32 [ 1, %3 ], [ %43, %42 ]
  %16 = icmp slt i32 %15, %7
  %17 = zext i1 %16 to i32
  %18 = shl i32 %15, %17
  %19 = ashr i32 %18, 1
  %20 = icmp sgt i32 %18, 1
  br i1 %20, label %21, label %45

21:                                               ; preds = %14
  %22 = trunc i64 %4 to i32
  %23 = add i32 %22, 1
  br label %52

24:                                               ; preds = %9, %42
  %25 = phi i32 [ %7, %9 ], [ %27, %42 ]
  %26 = phi i32 [ 1, %9 ], [ %43, %42 ]
  %27 = ashr i32 %25, 1
  tail call void @_Z7barrierj(i32 1) #5
  %28 = sext i32 %27 to i64
  %29 = icmp ult i64 %4, %28
  br i1 %29, label %30, label %42

30:                                               ; preds = %24
  %31 = mul i32 %26, %12
  %32 = add i32 %31, -1
  %33 = mul i32 %26, %13
  %34 = add i32 %33, -1
  %35 = sext i32 %32 to i64
  %36 = getelementptr inbounds i32, i32* %2, i64 %35
  %37 = load i32, i32* %36, align 4, !tbaa !3
  %38 = sext i32 %34 to i64
  %39 = getelementptr inbounds i32, i32* %2, i64 %38
  %40 = load i32, i32* %39, align 4, !tbaa !3
  %41 = add nsw i32 %40, %37
  store i32 %41, i32* %39, align 4, !tbaa !3
  br label %42

42:                                               ; preds = %30, %24
  %43 = shl i32 %26, 1
  %44 = icmp sgt i32 %25, 3
  br i1 %44, label %24, label %14

45:                                               ; preds = %71, %14
  tail call void @_Z7barrierj(i32 1) #5
  %46 = load i32, i32* %5, align 4, !tbaa !3
  %47 = load i32, i32* %0, align 4, !tbaa !3
  %48 = sub i32 %46, %1
  %49 = add i32 %48, %47
  tail call void @_Z7barrierj(i32 1) #5
  %50 = add i64 %6, -1
  %51 = icmp eq i64 %4, %50
  br i1 %51, label %74, label %78

52:                                               ; preds = %21, %71
  %53 = phi i32 [ %18, %21 ], [ %56, %71 ]
  %54 = phi i32 [ 0, %21 ], [ %72, %71 ]
  %55 = or i32 %54, 1
  %56 = ashr i32 %53, 1
  tail call void @_Z7barrierj(i32 1) #5
  %57 = sext i32 %55 to i64
  %58 = icmp ult i64 %4, %57
  br i1 %58, label %59, label %71

59:                                               ; preds = %52
  %60 = mul i32 %56, %23
  %61 = add i32 %60, -1
  %62 = ashr i32 %53, 2
  %63 = add nsw i32 %61, %62
  %64 = sext i32 %61 to i64
  %65 = getelementptr inbounds i32, i32* %2, i64 %64
  %66 = load i32, i32* %65, align 4, !tbaa !3
  %67 = sext i32 %63 to i64
  %68 = getelementptr inbounds i32, i32* %2, i64 %67
  %69 = load i32, i32* %68, align 4, !tbaa !3
  %70 = add nsw i32 %69, %66
  store i32 %70, i32* %68, align 4, !tbaa !3
  br label %71

71:                                               ; preds = %52, %59
  %72 = shl i32 %55, 1
  %73 = icmp slt i32 %72, %19
  br i1 %73, label %52, label %45

74:                                               ; preds = %45
  %75 = load i32, i32* %5, align 4, !tbaa !3
  %76 = load i32, i32* %0, align 4, !tbaa !3
  %77 = add nsw i32 %76, %75
  store i32 %77, i32* %0, align 4, !tbaa !3
  br label %78

78:                                               ; preds = %74, %45
  ret i32 %49
}

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @StreamCompaction_kernel(i32 %0, i32 %1, i32* nocapture %2, i32* nocapture %3, i32 %4, float %5, i32* nocapture %6, i32* nocapture readonly %7, i32* %8) local_unnamed_addr #3 !kernel_arg_addr_space !7 !kernel_arg_access_qual !8 !kernel_arg_type !9 !kernel_arg_base_type !9 !kernel_arg_type_qual !10 {
  %10 = fcmp oge float %5, 0.000000e+00
  %11 = fcmp ole float %5, 1.000000e+00
  %12 = and i1 %10, %11
  %13 = sitofp i32 %4 to float
  %14 = fmul float %13, %5
  %15 = fptosi float %14 to i32
  %16 = select i1 %12, i32 %15, i32 0
  %17 = tail call i64 @_Z12get_group_idj(i32 0) #4
  %18 = trunc i64 %17 to i32
  %19 = add i32 %16, %18
  %20 = icmp slt i32 %19, %4
  br i1 %20, label %21, label %29

21:                                               ; preds = %9
  %22 = tail call i64 @_Z12get_local_idj(i32 0) #4
  %23 = icmp eq i64 %22, 0
  %24 = trunc i64 %22 to i32
  %25 = shl i64 %22, 32
  %26 = ashr exact i64 %25, 32
  %27 = getelementptr inbounds i32, i32* %2, i64 %26
  %28 = icmp eq i32 %24, 0
  br label %30

29:                                               ; preds = %775, %9
  ret void

30:                                               ; preds = %21, %775
  %31 = phi i32 [ %19, %21 ], [ %778, %775 ]
  br i1 %23, label %32, label %33

32:                                               ; preds = %30
  store i32 0, i32* %3, align 4, !tbaa !3
  br label %33

33:                                               ; preds = %32, %30
  tail call void @_Z7barrierj(i32 1) #5
  %34 = sub nsw i32 %31, %16
  %35 = shl nsw i32 %34, 5
  %36 = sext i32 %35 to i64
  %37 = tail call i64 @_Z14get_local_sizej(i32 0) #4
  %38 = mul i64 %37, %36
  %39 = add i64 %38, %22
  %40 = trunc i64 %39 to i32
  %41 = icmp slt i32 %40, %0
  %42 = shl i64 %39, 32
  %43 = ashr exact i64 %42, 32
  br i1 %41, label %64, label %69

44:                                               ; preds = %56, %527
  br i1 %28, label %58, label %60

45:                                               ; preds = %527, %56
  %46 = phi i32 [ %47, %56 ], [ %530, %527 ]
  %47 = ashr i32 %46, 1
  %48 = icmp sgt i32 %47, %24
  br i1 %48, label %49, label %56

49:                                               ; preds = %45
  %50 = add nsw i32 %47, %24
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds i32, i32* %2, i64 %51
  %53 = load i32, i32* %52, align 4, !tbaa !3
  %54 = load i32, i32* %27, align 4, !tbaa !3
  %55 = add nsw i32 %54, %53
  store i32 %55, i32* %27, align 4, !tbaa !3
  br label %56

56:                                               ; preds = %49, %45
  tail call void @_Z7barrierj(i32 1) #5
  %57 = icmp sgt i32 %46, 3
  br i1 %57, label %45, label %44

58:                                               ; preds = %44
  %59 = load i32, i32* %2, align 4, !tbaa !3
  store i32 %59, i32* %3, align 4, !tbaa !3
  br label %60

60:                                               ; preds = %44, %58
  br i1 %23, label %61, label %88

61:                                               ; preds = %60
  %62 = sext i32 %31 to i64
  %63 = getelementptr inbounds i32, i32* %8, i64 %62
  br label %77

64:                                               ; preds = %33
  %65 = getelementptr inbounds i32, i32* %7, i64 %43
  %66 = load i32, i32* %65, align 4, !tbaa !3
  %67 = icmp ne i32 %66, %1
  %68 = zext i1 %67 to i32
  br label %69

69:                                               ; preds = %33, %64
  %70 = phi i32 [ %66, %64 ], [ %1, %33 ]
  %71 = phi i32 [ %68, %64 ], [ 0, %33 ]
  %72 = add i64 %43, %37
  %73 = trunc i64 %72 to i32
  %74 = icmp slt i32 %73, %0
  %75 = shl i64 %72, 32
  %76 = ashr exact i64 %75, 32
  br i1 %74, label %101, label %107

77:                                               ; preds = %61, %77
  %78 = tail call i32 @_Z10atomic_addPU8CLglobalVii(i32* %63, i32 0) #5
  %79 = icmp eq i32 %78, 0
  br i1 %79, label %77, label %80

80:                                               ; preds = %77
  %81 = add nsw i32 %31, 1
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds i32, i32* %8, i64 %82
  %84 = load i32, i32* %3, align 4, !tbaa !3
  %85 = add nsw i32 %84, %78
  %86 = tail call i32 @_Z10atomic_addPU8CLglobalVii(i32* %83, i32 %85) #5
  %87 = add nsw i32 %78, -1
  store i32 %87, i32* %3, align 4, !tbaa !3
  br label %88

88:                                               ; preds = %80, %60
  tail call void @_Z7barrierj(i32 3) #5
  %89 = icmp ne i32 %70, %1
  %90 = zext i1 %89 to i32
  %91 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %90, i32* %2) #6
  %92 = icmp eq i32 %70, %1
  br i1 %92, label %96, label %93

93:                                               ; preds = %88
  %94 = sext i32 %91 to i64
  %95 = getelementptr inbounds i32, i32* %6, i64 %94
  store i32 %70, i32* %95, align 4, !tbaa !3
  br label %96

96:                                               ; preds = %88, %93
  %97 = icmp ne i32 %108, %1
  %98 = zext i1 %97 to i32
  %99 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %98, i32* %2) #6
  %100 = icmp eq i32 %108, %1
  br i1 %100, label %535, label %532

101:                                              ; preds = %69
  %102 = getelementptr inbounds i32, i32* %7, i64 %76
  %103 = load i32, i32* %102, align 4, !tbaa !3
  %104 = icmp ne i32 %103, %1
  %105 = zext i1 %104 to i32
  %106 = add nuw nsw i32 %71, %105
  br label %107

107:                                              ; preds = %69, %101
  %108 = phi i32 [ %103, %101 ], [ %1, %69 ]
  %109 = phi i32 [ %106, %101 ], [ %71, %69 ]
  %110 = add i64 %76, %37
  %111 = trunc i64 %110 to i32
  %112 = icmp slt i32 %111, %0
  %113 = shl i64 %110, 32
  %114 = ashr exact i64 %113, 32
  br i1 %112, label %115, label %121

115:                                              ; preds = %107
  %116 = getelementptr inbounds i32, i32* %7, i64 %114
  %117 = load i32, i32* %116, align 4, !tbaa !3
  %118 = icmp ne i32 %117, %1
  %119 = zext i1 %118 to i32
  %120 = add nuw nsw i32 %109, %119
  br label %121

121:                                              ; preds = %107, %115
  %122 = phi i32 [ %117, %115 ], [ %1, %107 ]
  %123 = phi i32 [ %120, %115 ], [ %109, %107 ]
  %124 = add i64 %114, %37
  %125 = trunc i64 %124 to i32
  %126 = icmp slt i32 %125, %0
  %127 = shl i64 %124, 32
  %128 = ashr exact i64 %127, 32
  br i1 %126, label %129, label %135

129:                                              ; preds = %121
  %130 = getelementptr inbounds i32, i32* %7, i64 %128
  %131 = load i32, i32* %130, align 4, !tbaa !3
  %132 = icmp ne i32 %131, %1
  %133 = zext i1 %132 to i32
  %134 = add nuw nsw i32 %123, %133
  br label %135

135:                                              ; preds = %121, %129
  %136 = phi i32 [ %131, %129 ], [ %1, %121 ]
  %137 = phi i32 [ %134, %129 ], [ %123, %121 ]
  %138 = add i64 %128, %37
  %139 = trunc i64 %138 to i32
  %140 = icmp slt i32 %139, %0
  %141 = shl i64 %138, 32
  %142 = ashr exact i64 %141, 32
  br i1 %140, label %143, label %149

143:                                              ; preds = %135
  %144 = getelementptr inbounds i32, i32* %7, i64 %142
  %145 = load i32, i32* %144, align 4, !tbaa !3
  %146 = icmp ne i32 %145, %1
  %147 = zext i1 %146 to i32
  %148 = add nuw nsw i32 %137, %147
  br label %149

149:                                              ; preds = %135, %143
  %150 = phi i32 [ %145, %143 ], [ %1, %135 ]
  %151 = phi i32 [ %148, %143 ], [ %137, %135 ]
  %152 = add i64 %142, %37
  %153 = trunc i64 %152 to i32
  %154 = icmp slt i32 %153, %0
  %155 = shl i64 %152, 32
  %156 = ashr exact i64 %155, 32
  br i1 %154, label %157, label %163

157:                                              ; preds = %149
  %158 = getelementptr inbounds i32, i32* %7, i64 %156
  %159 = load i32, i32* %158, align 4, !tbaa !3
  %160 = icmp ne i32 %159, %1
  %161 = zext i1 %160 to i32
  %162 = add nuw nsw i32 %151, %161
  br label %163

163:                                              ; preds = %149, %157
  %164 = phi i32 [ %159, %157 ], [ %1, %149 ]
  %165 = phi i32 [ %162, %157 ], [ %151, %149 ]
  %166 = add i64 %156, %37
  %167 = trunc i64 %166 to i32
  %168 = icmp slt i32 %167, %0
  %169 = shl i64 %166, 32
  %170 = ashr exact i64 %169, 32
  br i1 %168, label %171, label %177

171:                                              ; preds = %163
  %172 = getelementptr inbounds i32, i32* %7, i64 %170
  %173 = load i32, i32* %172, align 4, !tbaa !3
  %174 = icmp ne i32 %173, %1
  %175 = zext i1 %174 to i32
  %176 = add nuw nsw i32 %165, %175
  br label %177

177:                                              ; preds = %163, %171
  %178 = phi i32 [ %173, %171 ], [ %1, %163 ]
  %179 = phi i32 [ %176, %171 ], [ %165, %163 ]
  %180 = add i64 %170, %37
  %181 = trunc i64 %180 to i32
  %182 = icmp slt i32 %181, %0
  %183 = shl i64 %180, 32
  %184 = ashr exact i64 %183, 32
  br i1 %182, label %185, label %191

185:                                              ; preds = %177
  %186 = getelementptr inbounds i32, i32* %7, i64 %184
  %187 = load i32, i32* %186, align 4, !tbaa !3
  %188 = icmp ne i32 %187, %1
  %189 = zext i1 %188 to i32
  %190 = add nuw nsw i32 %179, %189
  br label %191

191:                                              ; preds = %177, %185
  %192 = phi i32 [ %187, %185 ], [ %1, %177 ]
  %193 = phi i32 [ %190, %185 ], [ %179, %177 ]
  %194 = add i64 %184, %37
  %195 = trunc i64 %194 to i32
  %196 = icmp slt i32 %195, %0
  %197 = shl i64 %194, 32
  %198 = ashr exact i64 %197, 32
  br i1 %196, label %199, label %205

199:                                              ; preds = %191
  %200 = getelementptr inbounds i32, i32* %7, i64 %198
  %201 = load i32, i32* %200, align 4, !tbaa !3
  %202 = icmp ne i32 %201, %1
  %203 = zext i1 %202 to i32
  %204 = add nuw nsw i32 %193, %203
  br label %205

205:                                              ; preds = %191, %199
  %206 = phi i32 [ %201, %199 ], [ %1, %191 ]
  %207 = phi i32 [ %204, %199 ], [ %193, %191 ]
  %208 = add i64 %198, %37
  %209 = trunc i64 %208 to i32
  %210 = icmp slt i32 %209, %0
  %211 = shl i64 %208, 32
  %212 = ashr exact i64 %211, 32
  br i1 %210, label %213, label %219

213:                                              ; preds = %205
  %214 = getelementptr inbounds i32, i32* %7, i64 %212
  %215 = load i32, i32* %214, align 4, !tbaa !3
  %216 = icmp ne i32 %215, %1
  %217 = zext i1 %216 to i32
  %218 = add nuw nsw i32 %207, %217
  br label %219

219:                                              ; preds = %205, %213
  %220 = phi i32 [ %215, %213 ], [ %1, %205 ]
  %221 = phi i32 [ %218, %213 ], [ %207, %205 ]
  %222 = add i64 %212, %37
  %223 = trunc i64 %222 to i32
  %224 = icmp slt i32 %223, %0
  %225 = shl i64 %222, 32
  %226 = ashr exact i64 %225, 32
  br i1 %224, label %227, label %233

227:                                              ; preds = %219
  %228 = getelementptr inbounds i32, i32* %7, i64 %226
  %229 = load i32, i32* %228, align 4, !tbaa !3
  %230 = icmp ne i32 %229, %1
  %231 = zext i1 %230 to i32
  %232 = add nuw nsw i32 %221, %231
  br label %233

233:                                              ; preds = %219, %227
  %234 = phi i32 [ %229, %227 ], [ %1, %219 ]
  %235 = phi i32 [ %232, %227 ], [ %221, %219 ]
  %236 = add i64 %226, %37
  %237 = trunc i64 %236 to i32
  %238 = icmp slt i32 %237, %0
  %239 = shl i64 %236, 32
  %240 = ashr exact i64 %239, 32
  br i1 %238, label %241, label %247

241:                                              ; preds = %233
  %242 = getelementptr inbounds i32, i32* %7, i64 %240
  %243 = load i32, i32* %242, align 4, !tbaa !3
  %244 = icmp ne i32 %243, %1
  %245 = zext i1 %244 to i32
  %246 = add nuw nsw i32 %235, %245
  br label %247

247:                                              ; preds = %233, %241
  %248 = phi i32 [ %243, %241 ], [ %1, %233 ]
  %249 = phi i32 [ %246, %241 ], [ %235, %233 ]
  %250 = add i64 %240, %37
  %251 = trunc i64 %250 to i32
  %252 = icmp slt i32 %251, %0
  %253 = shl i64 %250, 32
  %254 = ashr exact i64 %253, 32
  br i1 %252, label %255, label %261

255:                                              ; preds = %247
  %256 = getelementptr inbounds i32, i32* %7, i64 %254
  %257 = load i32, i32* %256, align 4, !tbaa !3
  %258 = icmp ne i32 %257, %1
  %259 = zext i1 %258 to i32
  %260 = add nuw nsw i32 %249, %259
  br label %261

261:                                              ; preds = %247, %255
  %262 = phi i32 [ %257, %255 ], [ %1, %247 ]
  %263 = phi i32 [ %260, %255 ], [ %249, %247 ]
  %264 = add i64 %254, %37
  %265 = trunc i64 %264 to i32
  %266 = icmp slt i32 %265, %0
  %267 = shl i64 %264, 32
  %268 = ashr exact i64 %267, 32
  br i1 %266, label %269, label %275

269:                                              ; preds = %261
  %270 = getelementptr inbounds i32, i32* %7, i64 %268
  %271 = load i32, i32* %270, align 4, !tbaa !3
  %272 = icmp ne i32 %271, %1
  %273 = zext i1 %272 to i32
  %274 = add nuw nsw i32 %263, %273
  br label %275

275:                                              ; preds = %261, %269
  %276 = phi i32 [ %271, %269 ], [ %1, %261 ]
  %277 = phi i32 [ %274, %269 ], [ %263, %261 ]
  %278 = add i64 %268, %37
  %279 = trunc i64 %278 to i32
  %280 = icmp slt i32 %279, %0
  %281 = shl i64 %278, 32
  %282 = ashr exact i64 %281, 32
  br i1 %280, label %283, label %289

283:                                              ; preds = %275
  %284 = getelementptr inbounds i32, i32* %7, i64 %282
  %285 = load i32, i32* %284, align 4, !tbaa !3
  %286 = icmp ne i32 %285, %1
  %287 = zext i1 %286 to i32
  %288 = add nuw nsw i32 %277, %287
  br label %289

289:                                              ; preds = %275, %283
  %290 = phi i32 [ %285, %283 ], [ %1, %275 ]
  %291 = phi i32 [ %288, %283 ], [ %277, %275 ]
  %292 = add i64 %282, %37
  %293 = trunc i64 %292 to i32
  %294 = icmp slt i32 %293, %0
  %295 = shl i64 %292, 32
  %296 = ashr exact i64 %295, 32
  br i1 %294, label %297, label %303

297:                                              ; preds = %289
  %298 = getelementptr inbounds i32, i32* %7, i64 %296
  %299 = load i32, i32* %298, align 4, !tbaa !3
  %300 = icmp ne i32 %299, %1
  %301 = zext i1 %300 to i32
  %302 = add nuw nsw i32 %291, %301
  br label %303

303:                                              ; preds = %289, %297
  %304 = phi i32 [ %299, %297 ], [ %1, %289 ]
  %305 = phi i32 [ %302, %297 ], [ %291, %289 ]
  %306 = add i64 %296, %37
  %307 = trunc i64 %306 to i32
  %308 = icmp slt i32 %307, %0
  %309 = shl i64 %306, 32
  %310 = ashr exact i64 %309, 32
  br i1 %308, label %311, label %317

311:                                              ; preds = %303
  %312 = getelementptr inbounds i32, i32* %7, i64 %310
  %313 = load i32, i32* %312, align 4, !tbaa !3
  %314 = icmp ne i32 %313, %1
  %315 = zext i1 %314 to i32
  %316 = add nuw nsw i32 %305, %315
  br label %317

317:                                              ; preds = %303, %311
  %318 = phi i32 [ %313, %311 ], [ %1, %303 ]
  %319 = phi i32 [ %316, %311 ], [ %305, %303 ]
  %320 = add i64 %310, %37
  %321 = trunc i64 %320 to i32
  %322 = icmp slt i32 %321, %0
  %323 = shl i64 %320, 32
  %324 = ashr exact i64 %323, 32
  br i1 %322, label %325, label %331

325:                                              ; preds = %317
  %326 = getelementptr inbounds i32, i32* %7, i64 %324
  %327 = load i32, i32* %326, align 4, !tbaa !3
  %328 = icmp ne i32 %327, %1
  %329 = zext i1 %328 to i32
  %330 = add nuw nsw i32 %319, %329
  br label %331

331:                                              ; preds = %317, %325
  %332 = phi i32 [ %327, %325 ], [ %1, %317 ]
  %333 = phi i32 [ %330, %325 ], [ %319, %317 ]
  %334 = add i64 %324, %37
  %335 = trunc i64 %334 to i32
  %336 = icmp slt i32 %335, %0
  %337 = shl i64 %334, 32
  %338 = ashr exact i64 %337, 32
  br i1 %336, label %339, label %345

339:                                              ; preds = %331
  %340 = getelementptr inbounds i32, i32* %7, i64 %338
  %341 = load i32, i32* %340, align 4, !tbaa !3
  %342 = icmp ne i32 %341, %1
  %343 = zext i1 %342 to i32
  %344 = add nuw nsw i32 %333, %343
  br label %345

345:                                              ; preds = %331, %339
  %346 = phi i32 [ %341, %339 ], [ %1, %331 ]
  %347 = phi i32 [ %344, %339 ], [ %333, %331 ]
  %348 = add i64 %338, %37
  %349 = trunc i64 %348 to i32
  %350 = icmp slt i32 %349, %0
  %351 = shl i64 %348, 32
  %352 = ashr exact i64 %351, 32
  br i1 %350, label %353, label %359

353:                                              ; preds = %345
  %354 = getelementptr inbounds i32, i32* %7, i64 %352
  %355 = load i32, i32* %354, align 4, !tbaa !3
  %356 = icmp ne i32 %355, %1
  %357 = zext i1 %356 to i32
  %358 = add nuw nsw i32 %347, %357
  br label %359

359:                                              ; preds = %345, %353
  %360 = phi i32 [ %355, %353 ], [ %1, %345 ]
  %361 = phi i32 [ %358, %353 ], [ %347, %345 ]
  %362 = add i64 %352, %37
  %363 = trunc i64 %362 to i32
  %364 = icmp slt i32 %363, %0
  %365 = shl i64 %362, 32
  %366 = ashr exact i64 %365, 32
  br i1 %364, label %367, label %373

367:                                              ; preds = %359
  %368 = getelementptr inbounds i32, i32* %7, i64 %366
  %369 = load i32, i32* %368, align 4, !tbaa !3
  %370 = icmp ne i32 %369, %1
  %371 = zext i1 %370 to i32
  %372 = add nuw nsw i32 %361, %371
  br label %373

373:                                              ; preds = %359, %367
  %374 = phi i32 [ %369, %367 ], [ %1, %359 ]
  %375 = phi i32 [ %372, %367 ], [ %361, %359 ]
  %376 = add i64 %366, %37
  %377 = trunc i64 %376 to i32
  %378 = icmp slt i32 %377, %0
  %379 = shl i64 %376, 32
  %380 = ashr exact i64 %379, 32
  br i1 %378, label %381, label %387

381:                                              ; preds = %373
  %382 = getelementptr inbounds i32, i32* %7, i64 %380
  %383 = load i32, i32* %382, align 4, !tbaa !3
  %384 = icmp ne i32 %383, %1
  %385 = zext i1 %384 to i32
  %386 = add nuw nsw i32 %375, %385
  br label %387

387:                                              ; preds = %373, %381
  %388 = phi i32 [ %383, %381 ], [ %1, %373 ]
  %389 = phi i32 [ %386, %381 ], [ %375, %373 ]
  %390 = add i64 %380, %37
  %391 = trunc i64 %390 to i32
  %392 = icmp slt i32 %391, %0
  %393 = shl i64 %390, 32
  %394 = ashr exact i64 %393, 32
  br i1 %392, label %395, label %401

395:                                              ; preds = %387
  %396 = getelementptr inbounds i32, i32* %7, i64 %394
  %397 = load i32, i32* %396, align 4, !tbaa !3
  %398 = icmp ne i32 %397, %1
  %399 = zext i1 %398 to i32
  %400 = add nuw nsw i32 %389, %399
  br label %401

401:                                              ; preds = %387, %395
  %402 = phi i32 [ %397, %395 ], [ %1, %387 ]
  %403 = phi i32 [ %400, %395 ], [ %389, %387 ]
  %404 = add i64 %394, %37
  %405 = trunc i64 %404 to i32
  %406 = icmp slt i32 %405, %0
  %407 = shl i64 %404, 32
  %408 = ashr exact i64 %407, 32
  br i1 %406, label %409, label %415

409:                                              ; preds = %401
  %410 = getelementptr inbounds i32, i32* %7, i64 %408
  %411 = load i32, i32* %410, align 4, !tbaa !3
  %412 = icmp ne i32 %411, %1
  %413 = zext i1 %412 to i32
  %414 = add nuw nsw i32 %403, %413
  br label %415

415:                                              ; preds = %401, %409
  %416 = phi i32 [ %411, %409 ], [ %1, %401 ]
  %417 = phi i32 [ %414, %409 ], [ %403, %401 ]
  %418 = add i64 %408, %37
  %419 = trunc i64 %418 to i32
  %420 = icmp slt i32 %419, %0
  %421 = shl i64 %418, 32
  %422 = ashr exact i64 %421, 32
  br i1 %420, label %423, label %429

423:                                              ; preds = %415
  %424 = getelementptr inbounds i32, i32* %7, i64 %422
  %425 = load i32, i32* %424, align 4, !tbaa !3
  %426 = icmp ne i32 %425, %1
  %427 = zext i1 %426 to i32
  %428 = add nuw nsw i32 %417, %427
  br label %429

429:                                              ; preds = %415, %423
  %430 = phi i32 [ %425, %423 ], [ %1, %415 ]
  %431 = phi i32 [ %428, %423 ], [ %417, %415 ]
  %432 = add i64 %422, %37
  %433 = trunc i64 %432 to i32
  %434 = icmp slt i32 %433, %0
  %435 = shl i64 %432, 32
  %436 = ashr exact i64 %435, 32
  br i1 %434, label %437, label %443

437:                                              ; preds = %429
  %438 = getelementptr inbounds i32, i32* %7, i64 %436
  %439 = load i32, i32* %438, align 4, !tbaa !3
  %440 = icmp ne i32 %439, %1
  %441 = zext i1 %440 to i32
  %442 = add nuw nsw i32 %431, %441
  br label %443

443:                                              ; preds = %429, %437
  %444 = phi i32 [ %439, %437 ], [ %1, %429 ]
  %445 = phi i32 [ %442, %437 ], [ %431, %429 ]
  %446 = add i64 %436, %37
  %447 = trunc i64 %446 to i32
  %448 = icmp slt i32 %447, %0
  %449 = shl i64 %446, 32
  %450 = ashr exact i64 %449, 32
  br i1 %448, label %451, label %457

451:                                              ; preds = %443
  %452 = getelementptr inbounds i32, i32* %7, i64 %450
  %453 = load i32, i32* %452, align 4, !tbaa !3
  %454 = icmp ne i32 %453, %1
  %455 = zext i1 %454 to i32
  %456 = add nuw nsw i32 %445, %455
  br label %457

457:                                              ; preds = %443, %451
  %458 = phi i32 [ %453, %451 ], [ %1, %443 ]
  %459 = phi i32 [ %456, %451 ], [ %445, %443 ]
  %460 = add i64 %450, %37
  %461 = trunc i64 %460 to i32
  %462 = icmp slt i32 %461, %0
  %463 = shl i64 %460, 32
  %464 = ashr exact i64 %463, 32
  br i1 %462, label %465, label %471

465:                                              ; preds = %457
  %466 = getelementptr inbounds i32, i32* %7, i64 %464
  %467 = load i32, i32* %466, align 4, !tbaa !3
  %468 = icmp ne i32 %467, %1
  %469 = zext i1 %468 to i32
  %470 = add nuw nsw i32 %459, %469
  br label %471

471:                                              ; preds = %457, %465
  %472 = phi i32 [ %467, %465 ], [ %1, %457 ]
  %473 = phi i32 [ %470, %465 ], [ %459, %457 ]
  %474 = add i64 %464, %37
  %475 = trunc i64 %474 to i32
  %476 = icmp slt i32 %475, %0
  %477 = shl i64 %474, 32
  %478 = ashr exact i64 %477, 32
  br i1 %476, label %479, label %485

479:                                              ; preds = %471
  %480 = getelementptr inbounds i32, i32* %7, i64 %478
  %481 = load i32, i32* %480, align 4, !tbaa !3
  %482 = icmp ne i32 %481, %1
  %483 = zext i1 %482 to i32
  %484 = add nuw nsw i32 %473, %483
  br label %485

485:                                              ; preds = %471, %479
  %486 = phi i32 [ %481, %479 ], [ %1, %471 ]
  %487 = phi i32 [ %484, %479 ], [ %473, %471 ]
  %488 = add i64 %478, %37
  %489 = trunc i64 %488 to i32
  %490 = icmp slt i32 %489, %0
  %491 = shl i64 %488, 32
  %492 = ashr exact i64 %491, 32
  br i1 %490, label %493, label %499

493:                                              ; preds = %485
  %494 = getelementptr inbounds i32, i32* %7, i64 %492
  %495 = load i32, i32* %494, align 4, !tbaa !3
  %496 = icmp ne i32 %495, %1
  %497 = zext i1 %496 to i32
  %498 = add nuw nsw i32 %487, %497
  br label %499

499:                                              ; preds = %485, %493
  %500 = phi i32 [ %495, %493 ], [ %1, %485 ]
  %501 = phi i32 [ %498, %493 ], [ %487, %485 ]
  %502 = add i64 %492, %37
  %503 = trunc i64 %502 to i32
  %504 = icmp slt i32 %503, %0
  %505 = shl i64 %502, 32
  %506 = ashr exact i64 %505, 32
  br i1 %504, label %507, label %513

507:                                              ; preds = %499
  %508 = getelementptr inbounds i32, i32* %7, i64 %506
  %509 = load i32, i32* %508, align 4, !tbaa !3
  %510 = icmp ne i32 %509, %1
  %511 = zext i1 %510 to i32
  %512 = add nuw nsw i32 %501, %511
  br label %513

513:                                              ; preds = %499, %507
  %514 = phi i32 [ %509, %507 ], [ %1, %499 ]
  %515 = phi i32 [ %512, %507 ], [ %501, %499 ]
  %516 = add i64 %506, %37
  %517 = trunc i64 %516 to i32
  %518 = icmp slt i32 %517, %0
  br i1 %518, label %519, label %527

519:                                              ; preds = %513
  %520 = shl i64 %516, 32
  %521 = ashr exact i64 %520, 32
  %522 = getelementptr inbounds i32, i32* %7, i64 %521
  %523 = load i32, i32* %522, align 4, !tbaa !3
  %524 = icmp ne i32 %523, %1
  %525 = zext i1 %524 to i32
  %526 = add nuw nsw i32 %515, %525
  br label %527

527:                                              ; preds = %513, %519
  %528 = phi i32 [ %523, %519 ], [ %1, %513 ]
  %529 = phi i32 [ %526, %519 ], [ %515, %513 ]
  %530 = trunc i64 %37 to i32
  store i32 %529, i32* %27, align 4, !tbaa !3
  tail call void @_Z7barrierj(i32 1) #5
  %531 = icmp sgt i32 %530, 1
  br i1 %531, label %45, label %44

532:                                              ; preds = %96
  %533 = sext i32 %99 to i64
  %534 = getelementptr inbounds i32, i32* %6, i64 %533
  store i32 %108, i32* %534, align 4, !tbaa !3
  br label %535

535:                                              ; preds = %532, %96
  %536 = icmp ne i32 %122, %1
  %537 = zext i1 %536 to i32
  %538 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %537, i32* %2) #6
  %539 = icmp eq i32 %122, %1
  br i1 %539, label %543, label %540

540:                                              ; preds = %535
  %541 = sext i32 %538 to i64
  %542 = getelementptr inbounds i32, i32* %6, i64 %541
  store i32 %122, i32* %542, align 4, !tbaa !3
  br label %543

543:                                              ; preds = %540, %535
  %544 = icmp ne i32 %136, %1
  %545 = zext i1 %544 to i32
  %546 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %545, i32* %2) #6
  %547 = icmp eq i32 %136, %1
  br i1 %547, label %551, label %548

548:                                              ; preds = %543
  %549 = sext i32 %546 to i64
  %550 = getelementptr inbounds i32, i32* %6, i64 %549
  store i32 %136, i32* %550, align 4, !tbaa !3
  br label %551

551:                                              ; preds = %548, %543
  %552 = icmp ne i32 %150, %1
  %553 = zext i1 %552 to i32
  %554 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %553, i32* %2) #6
  %555 = icmp eq i32 %150, %1
  br i1 %555, label %559, label %556

556:                                              ; preds = %551
  %557 = sext i32 %554 to i64
  %558 = getelementptr inbounds i32, i32* %6, i64 %557
  store i32 %150, i32* %558, align 4, !tbaa !3
  br label %559

559:                                              ; preds = %556, %551
  %560 = icmp ne i32 %164, %1
  %561 = zext i1 %560 to i32
  %562 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %561, i32* %2) #6
  %563 = icmp eq i32 %164, %1
  br i1 %563, label %567, label %564

564:                                              ; preds = %559
  %565 = sext i32 %562 to i64
  %566 = getelementptr inbounds i32, i32* %6, i64 %565
  store i32 %164, i32* %566, align 4, !tbaa !3
  br label %567

567:                                              ; preds = %564, %559
  %568 = icmp ne i32 %178, %1
  %569 = zext i1 %568 to i32
  %570 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %569, i32* %2) #6
  %571 = icmp eq i32 %178, %1
  br i1 %571, label %575, label %572

572:                                              ; preds = %567
  %573 = sext i32 %570 to i64
  %574 = getelementptr inbounds i32, i32* %6, i64 %573
  store i32 %178, i32* %574, align 4, !tbaa !3
  br label %575

575:                                              ; preds = %572, %567
  %576 = icmp ne i32 %192, %1
  %577 = zext i1 %576 to i32
  %578 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %577, i32* %2) #6
  %579 = icmp eq i32 %192, %1
  br i1 %579, label %583, label %580

580:                                              ; preds = %575
  %581 = sext i32 %578 to i64
  %582 = getelementptr inbounds i32, i32* %6, i64 %581
  store i32 %192, i32* %582, align 4, !tbaa !3
  br label %583

583:                                              ; preds = %580, %575
  %584 = icmp ne i32 %206, %1
  %585 = zext i1 %584 to i32
  %586 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %585, i32* %2) #6
  %587 = icmp eq i32 %206, %1
  br i1 %587, label %591, label %588

588:                                              ; preds = %583
  %589 = sext i32 %586 to i64
  %590 = getelementptr inbounds i32, i32* %6, i64 %589
  store i32 %206, i32* %590, align 4, !tbaa !3
  br label %591

591:                                              ; preds = %588, %583
  %592 = icmp ne i32 %220, %1
  %593 = zext i1 %592 to i32
  %594 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %593, i32* %2) #6
  %595 = icmp eq i32 %220, %1
  br i1 %595, label %599, label %596

596:                                              ; preds = %591
  %597 = sext i32 %594 to i64
  %598 = getelementptr inbounds i32, i32* %6, i64 %597
  store i32 %220, i32* %598, align 4, !tbaa !3
  br label %599

599:                                              ; preds = %596, %591
  %600 = icmp ne i32 %234, %1
  %601 = zext i1 %600 to i32
  %602 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %601, i32* %2) #6
  %603 = icmp eq i32 %234, %1
  br i1 %603, label %607, label %604

604:                                              ; preds = %599
  %605 = sext i32 %602 to i64
  %606 = getelementptr inbounds i32, i32* %6, i64 %605
  store i32 %234, i32* %606, align 4, !tbaa !3
  br label %607

607:                                              ; preds = %604, %599
  %608 = icmp ne i32 %248, %1
  %609 = zext i1 %608 to i32
  %610 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %609, i32* %2) #6
  %611 = icmp eq i32 %248, %1
  br i1 %611, label %615, label %612

612:                                              ; preds = %607
  %613 = sext i32 %610 to i64
  %614 = getelementptr inbounds i32, i32* %6, i64 %613
  store i32 %248, i32* %614, align 4, !tbaa !3
  br label %615

615:                                              ; preds = %612, %607
  %616 = icmp ne i32 %262, %1
  %617 = zext i1 %616 to i32
  %618 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %617, i32* %2) #6
  %619 = icmp eq i32 %262, %1
  br i1 %619, label %623, label %620

620:                                              ; preds = %615
  %621 = sext i32 %618 to i64
  %622 = getelementptr inbounds i32, i32* %6, i64 %621
  store i32 %262, i32* %622, align 4, !tbaa !3
  br label %623

623:                                              ; preds = %620, %615
  %624 = icmp ne i32 %276, %1
  %625 = zext i1 %624 to i32
  %626 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %625, i32* %2) #6
  %627 = icmp eq i32 %276, %1
  br i1 %627, label %631, label %628

628:                                              ; preds = %623
  %629 = sext i32 %626 to i64
  %630 = getelementptr inbounds i32, i32* %6, i64 %629
  store i32 %276, i32* %630, align 4, !tbaa !3
  br label %631

631:                                              ; preds = %628, %623
  %632 = icmp ne i32 %290, %1
  %633 = zext i1 %632 to i32
  %634 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %633, i32* %2) #6
  %635 = icmp eq i32 %290, %1
  br i1 %635, label %639, label %636

636:                                              ; preds = %631
  %637 = sext i32 %634 to i64
  %638 = getelementptr inbounds i32, i32* %6, i64 %637
  store i32 %290, i32* %638, align 4, !tbaa !3
  br label %639

639:                                              ; preds = %636, %631
  %640 = icmp ne i32 %304, %1
  %641 = zext i1 %640 to i32
  %642 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %641, i32* %2) #6
  %643 = icmp eq i32 %304, %1
  br i1 %643, label %647, label %644

644:                                              ; preds = %639
  %645 = sext i32 %642 to i64
  %646 = getelementptr inbounds i32, i32* %6, i64 %645
  store i32 %304, i32* %646, align 4, !tbaa !3
  br label %647

647:                                              ; preds = %644, %639
  %648 = icmp ne i32 %318, %1
  %649 = zext i1 %648 to i32
  %650 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %649, i32* %2) #6
  %651 = icmp eq i32 %318, %1
  br i1 %651, label %655, label %652

652:                                              ; preds = %647
  %653 = sext i32 %650 to i64
  %654 = getelementptr inbounds i32, i32* %6, i64 %653
  store i32 %318, i32* %654, align 4, !tbaa !3
  br label %655

655:                                              ; preds = %652, %647
  %656 = icmp ne i32 %332, %1
  %657 = zext i1 %656 to i32
  %658 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %657, i32* %2) #6
  %659 = icmp eq i32 %332, %1
  br i1 %659, label %663, label %660

660:                                              ; preds = %655
  %661 = sext i32 %658 to i64
  %662 = getelementptr inbounds i32, i32* %6, i64 %661
  store i32 %332, i32* %662, align 4, !tbaa !3
  br label %663

663:                                              ; preds = %660, %655
  %664 = icmp ne i32 %346, %1
  %665 = zext i1 %664 to i32
  %666 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %665, i32* %2) #6
  %667 = icmp eq i32 %346, %1
  br i1 %667, label %671, label %668

668:                                              ; preds = %663
  %669 = sext i32 %666 to i64
  %670 = getelementptr inbounds i32, i32* %6, i64 %669
  store i32 %346, i32* %670, align 4, !tbaa !3
  br label %671

671:                                              ; preds = %668, %663
  %672 = icmp ne i32 %360, %1
  %673 = zext i1 %672 to i32
  %674 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %673, i32* %2) #6
  %675 = icmp eq i32 %360, %1
  br i1 %675, label %679, label %676

676:                                              ; preds = %671
  %677 = sext i32 %674 to i64
  %678 = getelementptr inbounds i32, i32* %6, i64 %677
  store i32 %360, i32* %678, align 4, !tbaa !3
  br label %679

679:                                              ; preds = %676, %671
  %680 = icmp ne i32 %374, %1
  %681 = zext i1 %680 to i32
  %682 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %681, i32* %2) #6
  %683 = icmp eq i32 %374, %1
  br i1 %683, label %687, label %684

684:                                              ; preds = %679
  %685 = sext i32 %682 to i64
  %686 = getelementptr inbounds i32, i32* %6, i64 %685
  store i32 %374, i32* %686, align 4, !tbaa !3
  br label %687

687:                                              ; preds = %684, %679
  %688 = icmp ne i32 %388, %1
  %689 = zext i1 %688 to i32
  %690 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %689, i32* %2) #6
  %691 = icmp eq i32 %388, %1
  br i1 %691, label %695, label %692

692:                                              ; preds = %687
  %693 = sext i32 %690 to i64
  %694 = getelementptr inbounds i32, i32* %6, i64 %693
  store i32 %388, i32* %694, align 4, !tbaa !3
  br label %695

695:                                              ; preds = %692, %687
  %696 = icmp ne i32 %402, %1
  %697 = zext i1 %696 to i32
  %698 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %697, i32* %2) #6
  %699 = icmp eq i32 %402, %1
  br i1 %699, label %703, label %700

700:                                              ; preds = %695
  %701 = sext i32 %698 to i64
  %702 = getelementptr inbounds i32, i32* %6, i64 %701
  store i32 %402, i32* %702, align 4, !tbaa !3
  br label %703

703:                                              ; preds = %700, %695
  %704 = icmp ne i32 %416, %1
  %705 = zext i1 %704 to i32
  %706 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %705, i32* %2) #6
  %707 = icmp eq i32 %416, %1
  br i1 %707, label %711, label %708

708:                                              ; preds = %703
  %709 = sext i32 %706 to i64
  %710 = getelementptr inbounds i32, i32* %6, i64 %709
  store i32 %416, i32* %710, align 4, !tbaa !3
  br label %711

711:                                              ; preds = %708, %703
  %712 = icmp ne i32 %430, %1
  %713 = zext i1 %712 to i32
  %714 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %713, i32* %2) #6
  %715 = icmp eq i32 %430, %1
  br i1 %715, label %719, label %716

716:                                              ; preds = %711
  %717 = sext i32 %714 to i64
  %718 = getelementptr inbounds i32, i32* %6, i64 %717
  store i32 %430, i32* %718, align 4, !tbaa !3
  br label %719

719:                                              ; preds = %716, %711
  %720 = icmp ne i32 %444, %1
  %721 = zext i1 %720 to i32
  %722 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %721, i32* %2) #6
  %723 = icmp eq i32 %444, %1
  br i1 %723, label %727, label %724

724:                                              ; preds = %719
  %725 = sext i32 %722 to i64
  %726 = getelementptr inbounds i32, i32* %6, i64 %725
  store i32 %444, i32* %726, align 4, !tbaa !3
  br label %727

727:                                              ; preds = %724, %719
  %728 = icmp ne i32 %458, %1
  %729 = zext i1 %728 to i32
  %730 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %729, i32* %2) #6
  %731 = icmp eq i32 %458, %1
  br i1 %731, label %735, label %732

732:                                              ; preds = %727
  %733 = sext i32 %730 to i64
  %734 = getelementptr inbounds i32, i32* %6, i64 %733
  store i32 %458, i32* %734, align 4, !tbaa !3
  br label %735

735:                                              ; preds = %732, %727
  %736 = icmp ne i32 %472, %1
  %737 = zext i1 %736 to i32
  %738 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %737, i32* %2) #6
  %739 = icmp eq i32 %472, %1
  br i1 %739, label %743, label %740

740:                                              ; preds = %735
  %741 = sext i32 %738 to i64
  %742 = getelementptr inbounds i32, i32* %6, i64 %741
  store i32 %472, i32* %742, align 4, !tbaa !3
  br label %743

743:                                              ; preds = %740, %735
  %744 = icmp ne i32 %486, %1
  %745 = zext i1 %744 to i32
  %746 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %745, i32* %2) #6
  %747 = icmp eq i32 %486, %1
  br i1 %747, label %751, label %748

748:                                              ; preds = %743
  %749 = sext i32 %746 to i64
  %750 = getelementptr inbounds i32, i32* %6, i64 %749
  store i32 %486, i32* %750, align 4, !tbaa !3
  br label %751

751:                                              ; preds = %748, %743
  %752 = icmp ne i32 %500, %1
  %753 = zext i1 %752 to i32
  %754 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %753, i32* %2) #6
  %755 = icmp eq i32 %500, %1
  br i1 %755, label %759, label %756

756:                                              ; preds = %751
  %757 = sext i32 %754 to i64
  %758 = getelementptr inbounds i32, i32* %6, i64 %757
  store i32 %500, i32* %758, align 4, !tbaa !3
  br label %759

759:                                              ; preds = %756, %751
  %760 = icmp ne i32 %514, %1
  %761 = zext i1 %760 to i32
  %762 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %761, i32* %2) #6
  %763 = icmp eq i32 %514, %1
  br i1 %763, label %767, label %764

764:                                              ; preds = %759
  %765 = sext i32 %762 to i64
  %766 = getelementptr inbounds i32, i32* %6, i64 %765
  store i32 %514, i32* %766, align 4, !tbaa !3
  br label %767

767:                                              ; preds = %764, %759
  %768 = icmp ne i32 %528, %1
  %769 = zext i1 %768 to i32
  %770 = tail call i32 @block_binary_prefix_sums(i32* %3, i32 %769, i32* %2) #6
  %771 = icmp eq i32 %528, %1
  br i1 %771, label %775, label %772

772:                                              ; preds = %767
  %773 = sext i32 %770 to i64
  %774 = getelementptr inbounds i32, i32* %6, i64 %773
  store i32 %528, i32* %774, align 4, !tbaa !3
  br label %775

775:                                              ; preds = %772, %767
  %776 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
  %777 = trunc i64 %776 to i32
  %778 = add i32 %31, %777
  %779 = icmp slt i32 %778, %4
  br i1 %779, label %30, label %29
}

; Function Attrs: convergent
declare dso_local i32 @_Z10atomic_addPU8CLglobalVii(i32*, i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_group_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_num_groupsj(i32) local_unnamed_addr #1

attributes #0 = { convergent nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { convergent nounwind readnone }
attributes #5 = { convergent nounwind }
attributes #6 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{!"clang version 10.0.0-4ubuntu1 "}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{i32 0, i32 0, i32 3, i32 3, i32 0, i32 0, i32 1, i32 1, i32 1}
!8 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!9 = !{!"int", !"int", !"int*", !"int*", !"int", !"float", !"int*", !"int*", !"int*"}
!10 = !{!"", !"", !"", !"", !"", !"", !"", !"", !""}
