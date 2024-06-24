; ModuleID = './kernels/pad.cl'
source_filename = "./kernels/pad.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @Padding_kernel(i32 %0, i32 %1, i32 %2, i32 %3, float %4, double* nocapture %5, double* nocapture readonly %6, i32* %7) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %9 = fcmp ult float %4, 0.000000e+00
  %10 = fcmp ugt float %4, 1.000000e+00
  %11 = or i1 %9, %10
  %12 = sitofp i32 %3 to float
  %13 = fmul float %12, %4
  %14 = fptosi float %13 to i32
  %15 = select i1 %11, i32 0, i32 %14
  %16 = add nsw i32 %2, %0
  %17 = mul nsw i32 %16, %1
  %18 = sext i32 %17 to i64
  %19 = tail call i64 @_Z14get_local_sizej(i32 0) #3
  %20 = shl i64 %19, 5
  %21 = add nsw i64 %18, -1
  %22 = add i64 %21, %20
  %23 = tail call i64 @_Z12get_group_idj(i32 0) #3
  %24 = trunc i64 %23 to i32
  %25 = add i32 %15, %24
  %26 = icmp slt i32 %25, %3
  br i1 %26, label %27, label %36

27:                                               ; preds = %8
  %28 = urem i64 %22, %20
  %29 = sub i64 %22, %28
  %30 = shl i64 %29, 32
  %31 = add i64 %30, -4294967296
  %32 = lshr exact i64 %31, 32
  %33 = tail call i64 @_Z12get_local_idj(i32 0) #3
  %34 = icmp eq i64 %33, 0
  %35 = trunc i64 %19 to i32
  br label %37

36:                                               ; preds = %977, %8
  ret void

37:                                               ; preds = %27, %977
  %38 = phi i32 [ %25, %27 ], [ %980, %977 ]
  %39 = shl nsw i32 %38, 5
  %40 = sext i32 %39 to i64
  %41 = mul i64 %19, %40
  %42 = sub i64 %32, %41
  %43 = sub i64 %42, %33
  %44 = trunc i64 %43 to i32
  %45 = sdiv i32 %44, %16
  %46 = srem i32 %44, %16
  %47 = mul nsw i32 %45, %0
  %48 = add nsw i32 %47, %46
  %49 = icmp sgt i32 %48, -1
  %50 = icmp slt i32 %46, %0
  %51 = and i1 %50, %49
  %52 = icmp slt i32 %48, %17
  %53 = and i1 %52, %51
  br i1 %53, label %57, label %62

54:                                               ; preds = %701
  %55 = sext i32 %38 to i64
  %56 = getelementptr inbounds i32, i32* %7, i64 %55
  br label %77

57:                                               ; preds = %37
  %58 = zext i32 %48 to i64
  %59 = getelementptr inbounds double, double* %6, i64 %58
  %60 = bitcast double* %59 to i64*
  %61 = load i64, i64* %60, align 8, !tbaa !7
  br label %62

62:                                               ; preds = %37, %57
  %63 = phi i64 [ %61, %57 ], [ 0, %37 ]
  %64 = shl i64 %43, 32
  %65 = ashr exact i64 %64, 32
  %66 = sub i64 %65, %19
  %67 = trunc i64 %66 to i32
  %68 = sdiv i32 %67, %16
  %69 = srem i32 %67, %16
  %70 = mul nsw i32 %68, %0
  %71 = add nsw i32 %70, %69
  %72 = icmp sgt i32 %71, -1
  %73 = icmp slt i32 %69, %0
  %74 = and i1 %73, %72
  %75 = icmp slt i32 %71, %17
  %76 = and i1 %75, %74
  br i1 %76, label %98, label %103

77:                                               ; preds = %54, %77
  %78 = tail call i32 @_Z10atomic_addPU8CLglobalVii(i32* %56, i32 0) #4
  %79 = icmp eq i32 %78, 0
  br i1 %79, label %77, label %80

80:                                               ; preds = %77
  %81 = add nsw i32 %38, 1
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds i32, i32* %7, i64 %82
  %84 = tail call i32 @_Z10atomic_addPU8CLglobalVii(i32* %83, i32 1) #4
  br label %85

85:                                               ; preds = %80, %701
  tail call void @_Z7barrierj(i32 3) #4
  %86 = icmp sgt i32 %44, -1
  %87 = icmp sgt i32 %17, %44
  %88 = and i1 %86, %87
  br i1 %88, label %89, label %93

89:                                               ; preds = %85
  %90 = and i64 %43, 4294967295
  %91 = getelementptr inbounds double, double* %5, i64 %90
  %92 = bitcast double* %91 to i64*
  store i64 %63, i64* %92, align 8, !tbaa !7
  br label %93

93:                                               ; preds = %89, %85
  %94 = sub i32 %44, %35
  %95 = icmp sgt i32 %94, -1
  %96 = icmp slt i32 %94, %17
  %97 = and i1 %95, %96
  br i1 %97, label %703, label %707

98:                                               ; preds = %62
  %99 = zext i32 %71 to i64
  %100 = getelementptr inbounds double, double* %6, i64 %99
  %101 = bitcast double* %100 to i64*
  %102 = load i64, i64* %101, align 8, !tbaa !7
  br label %103

103:                                              ; preds = %62, %98
  %104 = phi i64 [ %102, %98 ], [ 0, %62 ]
  %105 = shl i64 %66, 32
  %106 = ashr exact i64 %105, 32
  %107 = sub i64 %106, %19
  %108 = trunc i64 %107 to i32
  %109 = sdiv i32 %108, %16
  %110 = srem i32 %108, %16
  %111 = mul nsw i32 %109, %0
  %112 = add nsw i32 %111, %110
  %113 = icmp sgt i32 %112, -1
  %114 = icmp slt i32 %110, %0
  %115 = and i1 %114, %113
  %116 = icmp slt i32 %112, %17
  %117 = and i1 %116, %115
  br i1 %117, label %118, label %123

118:                                              ; preds = %103
  %119 = zext i32 %112 to i64
  %120 = getelementptr inbounds double, double* %6, i64 %119
  %121 = bitcast double* %120 to i64*
  %122 = load i64, i64* %121, align 8, !tbaa !7
  br label %123

123:                                              ; preds = %103, %118
  %124 = phi i64 [ %122, %118 ], [ 0, %103 ]
  %125 = shl i64 %107, 32
  %126 = ashr exact i64 %125, 32
  %127 = sub i64 %126, %19
  %128 = trunc i64 %127 to i32
  %129 = sdiv i32 %128, %16
  %130 = srem i32 %128, %16
  %131 = mul nsw i32 %129, %0
  %132 = add nsw i32 %131, %130
  %133 = icmp sgt i32 %132, -1
  %134 = icmp slt i32 %130, %0
  %135 = and i1 %134, %133
  %136 = icmp slt i32 %132, %17
  %137 = and i1 %136, %135
  br i1 %137, label %138, label %143

138:                                              ; preds = %123
  %139 = zext i32 %132 to i64
  %140 = getelementptr inbounds double, double* %6, i64 %139
  %141 = bitcast double* %140 to i64*
  %142 = load i64, i64* %141, align 8, !tbaa !7
  br label %143

143:                                              ; preds = %123, %138
  %144 = phi i64 [ %142, %138 ], [ 0, %123 ]
  %145 = shl i64 %127, 32
  %146 = ashr exact i64 %145, 32
  %147 = sub i64 %146, %19
  %148 = trunc i64 %147 to i32
  %149 = sdiv i32 %148, %16
  %150 = srem i32 %148, %16
  %151 = mul nsw i32 %149, %0
  %152 = add nsw i32 %151, %150
  %153 = icmp sgt i32 %152, -1
  %154 = icmp slt i32 %150, %0
  %155 = and i1 %154, %153
  %156 = icmp slt i32 %152, %17
  %157 = and i1 %156, %155
  br i1 %157, label %158, label %163

158:                                              ; preds = %143
  %159 = zext i32 %152 to i64
  %160 = getelementptr inbounds double, double* %6, i64 %159
  %161 = bitcast double* %160 to i64*
  %162 = load i64, i64* %161, align 8, !tbaa !7
  br label %163

163:                                              ; preds = %143, %158
  %164 = phi i64 [ %162, %158 ], [ 0, %143 ]
  %165 = shl i64 %147, 32
  %166 = ashr exact i64 %165, 32
  %167 = sub i64 %166, %19
  %168 = trunc i64 %167 to i32
  %169 = sdiv i32 %168, %16
  %170 = srem i32 %168, %16
  %171 = mul nsw i32 %169, %0
  %172 = add nsw i32 %171, %170
  %173 = icmp sgt i32 %172, -1
  %174 = icmp slt i32 %170, %0
  %175 = and i1 %174, %173
  %176 = icmp slt i32 %172, %17
  %177 = and i1 %176, %175
  br i1 %177, label %178, label %183

178:                                              ; preds = %163
  %179 = zext i32 %172 to i64
  %180 = getelementptr inbounds double, double* %6, i64 %179
  %181 = bitcast double* %180 to i64*
  %182 = load i64, i64* %181, align 8, !tbaa !7
  br label %183

183:                                              ; preds = %163, %178
  %184 = phi i64 [ %182, %178 ], [ 0, %163 ]
  %185 = shl i64 %167, 32
  %186 = ashr exact i64 %185, 32
  %187 = sub i64 %186, %19
  %188 = trunc i64 %187 to i32
  %189 = sdiv i32 %188, %16
  %190 = srem i32 %188, %16
  %191 = mul nsw i32 %189, %0
  %192 = add nsw i32 %191, %190
  %193 = icmp sgt i32 %192, -1
  %194 = icmp slt i32 %190, %0
  %195 = and i1 %194, %193
  %196 = icmp slt i32 %192, %17
  %197 = and i1 %196, %195
  br i1 %197, label %198, label %203

198:                                              ; preds = %183
  %199 = zext i32 %192 to i64
  %200 = getelementptr inbounds double, double* %6, i64 %199
  %201 = bitcast double* %200 to i64*
  %202 = load i64, i64* %201, align 8, !tbaa !7
  br label %203

203:                                              ; preds = %183, %198
  %204 = phi i64 [ %202, %198 ], [ 0, %183 ]
  %205 = shl i64 %187, 32
  %206 = ashr exact i64 %205, 32
  %207 = sub i64 %206, %19
  %208 = trunc i64 %207 to i32
  %209 = sdiv i32 %208, %16
  %210 = srem i32 %208, %16
  %211 = mul nsw i32 %209, %0
  %212 = add nsw i32 %211, %210
  %213 = icmp sgt i32 %212, -1
  %214 = icmp slt i32 %210, %0
  %215 = and i1 %214, %213
  %216 = icmp slt i32 %212, %17
  %217 = and i1 %216, %215
  br i1 %217, label %218, label %223

218:                                              ; preds = %203
  %219 = zext i32 %212 to i64
  %220 = getelementptr inbounds double, double* %6, i64 %219
  %221 = bitcast double* %220 to i64*
  %222 = load i64, i64* %221, align 8, !tbaa !7
  br label %223

223:                                              ; preds = %203, %218
  %224 = phi i64 [ %222, %218 ], [ 0, %203 ]
  %225 = shl i64 %207, 32
  %226 = ashr exact i64 %225, 32
  %227 = sub i64 %226, %19
  %228 = trunc i64 %227 to i32
  %229 = sdiv i32 %228, %16
  %230 = srem i32 %228, %16
  %231 = mul nsw i32 %229, %0
  %232 = add nsw i32 %231, %230
  %233 = icmp sgt i32 %232, -1
  %234 = icmp slt i32 %230, %0
  %235 = and i1 %234, %233
  %236 = icmp slt i32 %232, %17
  %237 = and i1 %236, %235
  br i1 %237, label %238, label %243

238:                                              ; preds = %223
  %239 = zext i32 %232 to i64
  %240 = getelementptr inbounds double, double* %6, i64 %239
  %241 = bitcast double* %240 to i64*
  %242 = load i64, i64* %241, align 8, !tbaa !7
  br label %243

243:                                              ; preds = %223, %238
  %244 = phi i64 [ %242, %238 ], [ 0, %223 ]
  %245 = shl i64 %227, 32
  %246 = ashr exact i64 %245, 32
  %247 = sub i64 %246, %19
  %248 = trunc i64 %247 to i32
  %249 = sdiv i32 %248, %16
  %250 = srem i32 %248, %16
  %251 = mul nsw i32 %249, %0
  %252 = add nsw i32 %251, %250
  %253 = icmp sgt i32 %252, -1
  %254 = icmp slt i32 %250, %0
  %255 = and i1 %254, %253
  %256 = icmp slt i32 %252, %17
  %257 = and i1 %256, %255
  br i1 %257, label %258, label %263

258:                                              ; preds = %243
  %259 = zext i32 %252 to i64
  %260 = getelementptr inbounds double, double* %6, i64 %259
  %261 = bitcast double* %260 to i64*
  %262 = load i64, i64* %261, align 8, !tbaa !7
  br label %263

263:                                              ; preds = %243, %258
  %264 = phi i64 [ %262, %258 ], [ 0, %243 ]
  %265 = shl i64 %247, 32
  %266 = ashr exact i64 %265, 32
  %267 = sub i64 %266, %19
  %268 = trunc i64 %267 to i32
  %269 = sdiv i32 %268, %16
  %270 = srem i32 %268, %16
  %271 = mul nsw i32 %269, %0
  %272 = add nsw i32 %271, %270
  %273 = icmp sgt i32 %272, -1
  %274 = icmp slt i32 %270, %0
  %275 = and i1 %274, %273
  %276 = icmp slt i32 %272, %17
  %277 = and i1 %276, %275
  br i1 %277, label %278, label %283

278:                                              ; preds = %263
  %279 = zext i32 %272 to i64
  %280 = getelementptr inbounds double, double* %6, i64 %279
  %281 = bitcast double* %280 to i64*
  %282 = load i64, i64* %281, align 8, !tbaa !7
  br label %283

283:                                              ; preds = %263, %278
  %284 = phi i64 [ %282, %278 ], [ 0, %263 ]
  %285 = shl i64 %267, 32
  %286 = ashr exact i64 %285, 32
  %287 = sub i64 %286, %19
  %288 = trunc i64 %287 to i32
  %289 = sdiv i32 %288, %16
  %290 = srem i32 %288, %16
  %291 = mul nsw i32 %289, %0
  %292 = add nsw i32 %291, %290
  %293 = icmp sgt i32 %292, -1
  %294 = icmp slt i32 %290, %0
  %295 = and i1 %294, %293
  %296 = icmp slt i32 %292, %17
  %297 = and i1 %296, %295
  br i1 %297, label %298, label %303

298:                                              ; preds = %283
  %299 = zext i32 %292 to i64
  %300 = getelementptr inbounds double, double* %6, i64 %299
  %301 = bitcast double* %300 to i64*
  %302 = load i64, i64* %301, align 8, !tbaa !7
  br label %303

303:                                              ; preds = %283, %298
  %304 = phi i64 [ %302, %298 ], [ 0, %283 ]
  %305 = shl i64 %287, 32
  %306 = ashr exact i64 %305, 32
  %307 = sub i64 %306, %19
  %308 = trunc i64 %307 to i32
  %309 = sdiv i32 %308, %16
  %310 = srem i32 %308, %16
  %311 = mul nsw i32 %309, %0
  %312 = add nsw i32 %311, %310
  %313 = icmp sgt i32 %312, -1
  %314 = icmp slt i32 %310, %0
  %315 = and i1 %314, %313
  %316 = icmp slt i32 %312, %17
  %317 = and i1 %316, %315
  br i1 %317, label %318, label %323

318:                                              ; preds = %303
  %319 = zext i32 %312 to i64
  %320 = getelementptr inbounds double, double* %6, i64 %319
  %321 = bitcast double* %320 to i64*
  %322 = load i64, i64* %321, align 8, !tbaa !7
  br label %323

323:                                              ; preds = %303, %318
  %324 = phi i64 [ %322, %318 ], [ 0, %303 ]
  %325 = shl i64 %307, 32
  %326 = ashr exact i64 %325, 32
  %327 = sub i64 %326, %19
  %328 = trunc i64 %327 to i32
  %329 = sdiv i32 %328, %16
  %330 = srem i32 %328, %16
  %331 = mul nsw i32 %329, %0
  %332 = add nsw i32 %331, %330
  %333 = icmp sgt i32 %332, -1
  %334 = icmp slt i32 %330, %0
  %335 = and i1 %334, %333
  %336 = icmp slt i32 %332, %17
  %337 = and i1 %336, %335
  br i1 %337, label %338, label %343

338:                                              ; preds = %323
  %339 = zext i32 %332 to i64
  %340 = getelementptr inbounds double, double* %6, i64 %339
  %341 = bitcast double* %340 to i64*
  %342 = load i64, i64* %341, align 8, !tbaa !7
  br label %343

343:                                              ; preds = %323, %338
  %344 = phi i64 [ %342, %338 ], [ 0, %323 ]
  %345 = shl i64 %327, 32
  %346 = ashr exact i64 %345, 32
  %347 = sub i64 %346, %19
  %348 = trunc i64 %347 to i32
  %349 = sdiv i32 %348, %16
  %350 = srem i32 %348, %16
  %351 = mul nsw i32 %349, %0
  %352 = add nsw i32 %351, %350
  %353 = icmp sgt i32 %352, -1
  %354 = icmp slt i32 %350, %0
  %355 = and i1 %354, %353
  %356 = icmp slt i32 %352, %17
  %357 = and i1 %356, %355
  br i1 %357, label %358, label %363

358:                                              ; preds = %343
  %359 = zext i32 %352 to i64
  %360 = getelementptr inbounds double, double* %6, i64 %359
  %361 = bitcast double* %360 to i64*
  %362 = load i64, i64* %361, align 8, !tbaa !7
  br label %363

363:                                              ; preds = %343, %358
  %364 = phi i64 [ %362, %358 ], [ 0, %343 ]
  %365 = shl i64 %347, 32
  %366 = ashr exact i64 %365, 32
  %367 = sub i64 %366, %19
  %368 = trunc i64 %367 to i32
  %369 = sdiv i32 %368, %16
  %370 = srem i32 %368, %16
  %371 = mul nsw i32 %369, %0
  %372 = add nsw i32 %371, %370
  %373 = icmp sgt i32 %372, -1
  %374 = icmp slt i32 %370, %0
  %375 = and i1 %374, %373
  %376 = icmp slt i32 %372, %17
  %377 = and i1 %376, %375
  br i1 %377, label %378, label %383

378:                                              ; preds = %363
  %379 = zext i32 %372 to i64
  %380 = getelementptr inbounds double, double* %6, i64 %379
  %381 = bitcast double* %380 to i64*
  %382 = load i64, i64* %381, align 8, !tbaa !7
  br label %383

383:                                              ; preds = %363, %378
  %384 = phi i64 [ %382, %378 ], [ 0, %363 ]
  %385 = shl i64 %367, 32
  %386 = ashr exact i64 %385, 32
  %387 = sub i64 %386, %19
  %388 = trunc i64 %387 to i32
  %389 = sdiv i32 %388, %16
  %390 = srem i32 %388, %16
  %391 = mul nsw i32 %389, %0
  %392 = add nsw i32 %391, %390
  %393 = icmp sgt i32 %392, -1
  %394 = icmp slt i32 %390, %0
  %395 = and i1 %394, %393
  %396 = icmp slt i32 %392, %17
  %397 = and i1 %396, %395
  br i1 %397, label %398, label %403

398:                                              ; preds = %383
  %399 = zext i32 %392 to i64
  %400 = getelementptr inbounds double, double* %6, i64 %399
  %401 = bitcast double* %400 to i64*
  %402 = load i64, i64* %401, align 8, !tbaa !7
  br label %403

403:                                              ; preds = %383, %398
  %404 = phi i64 [ %402, %398 ], [ 0, %383 ]
  %405 = shl i64 %387, 32
  %406 = ashr exact i64 %405, 32
  %407 = sub i64 %406, %19
  %408 = trunc i64 %407 to i32
  %409 = sdiv i32 %408, %16
  %410 = srem i32 %408, %16
  %411 = mul nsw i32 %409, %0
  %412 = add nsw i32 %411, %410
  %413 = icmp sgt i32 %412, -1
  %414 = icmp slt i32 %410, %0
  %415 = and i1 %414, %413
  %416 = icmp slt i32 %412, %17
  %417 = and i1 %416, %415
  br i1 %417, label %418, label %423

418:                                              ; preds = %403
  %419 = zext i32 %412 to i64
  %420 = getelementptr inbounds double, double* %6, i64 %419
  %421 = bitcast double* %420 to i64*
  %422 = load i64, i64* %421, align 8, !tbaa !7
  br label %423

423:                                              ; preds = %403, %418
  %424 = phi i64 [ %422, %418 ], [ 0, %403 ]
  %425 = shl i64 %407, 32
  %426 = ashr exact i64 %425, 32
  %427 = sub i64 %426, %19
  %428 = trunc i64 %427 to i32
  %429 = sdiv i32 %428, %16
  %430 = srem i32 %428, %16
  %431 = mul nsw i32 %429, %0
  %432 = add nsw i32 %431, %430
  %433 = icmp sgt i32 %432, -1
  %434 = icmp slt i32 %430, %0
  %435 = and i1 %434, %433
  %436 = icmp slt i32 %432, %17
  %437 = and i1 %436, %435
  br i1 %437, label %438, label %443

438:                                              ; preds = %423
  %439 = zext i32 %432 to i64
  %440 = getelementptr inbounds double, double* %6, i64 %439
  %441 = bitcast double* %440 to i64*
  %442 = load i64, i64* %441, align 8, !tbaa !7
  br label %443

443:                                              ; preds = %423, %438
  %444 = phi i64 [ %442, %438 ], [ 0, %423 ]
  %445 = shl i64 %427, 32
  %446 = ashr exact i64 %445, 32
  %447 = sub i64 %446, %19
  %448 = trunc i64 %447 to i32
  %449 = sdiv i32 %448, %16
  %450 = srem i32 %448, %16
  %451 = mul nsw i32 %449, %0
  %452 = add nsw i32 %451, %450
  %453 = icmp sgt i32 %452, -1
  %454 = icmp slt i32 %450, %0
  %455 = and i1 %454, %453
  %456 = icmp slt i32 %452, %17
  %457 = and i1 %456, %455
  br i1 %457, label %458, label %463

458:                                              ; preds = %443
  %459 = zext i32 %452 to i64
  %460 = getelementptr inbounds double, double* %6, i64 %459
  %461 = bitcast double* %460 to i64*
  %462 = load i64, i64* %461, align 8, !tbaa !7
  br label %463

463:                                              ; preds = %443, %458
  %464 = phi i64 [ %462, %458 ], [ 0, %443 ]
  %465 = shl i64 %447, 32
  %466 = ashr exact i64 %465, 32
  %467 = sub i64 %466, %19
  %468 = trunc i64 %467 to i32
  %469 = sdiv i32 %468, %16
  %470 = srem i32 %468, %16
  %471 = mul nsw i32 %469, %0
  %472 = add nsw i32 %471, %470
  %473 = icmp sgt i32 %472, -1
  %474 = icmp slt i32 %470, %0
  %475 = and i1 %474, %473
  %476 = icmp slt i32 %472, %17
  %477 = and i1 %476, %475
  br i1 %477, label %478, label %483

478:                                              ; preds = %463
  %479 = zext i32 %472 to i64
  %480 = getelementptr inbounds double, double* %6, i64 %479
  %481 = bitcast double* %480 to i64*
  %482 = load i64, i64* %481, align 8, !tbaa !7
  br label %483

483:                                              ; preds = %463, %478
  %484 = phi i64 [ %482, %478 ], [ 0, %463 ]
  %485 = shl i64 %467, 32
  %486 = ashr exact i64 %485, 32
  %487 = sub i64 %486, %19
  %488 = trunc i64 %487 to i32
  %489 = sdiv i32 %488, %16
  %490 = srem i32 %488, %16
  %491 = mul nsw i32 %489, %0
  %492 = add nsw i32 %491, %490
  %493 = icmp sgt i32 %492, -1
  %494 = icmp slt i32 %490, %0
  %495 = and i1 %494, %493
  %496 = icmp slt i32 %492, %17
  %497 = and i1 %496, %495
  br i1 %497, label %498, label %503

498:                                              ; preds = %483
  %499 = zext i32 %492 to i64
  %500 = getelementptr inbounds double, double* %6, i64 %499
  %501 = bitcast double* %500 to i64*
  %502 = load i64, i64* %501, align 8, !tbaa !7
  br label %503

503:                                              ; preds = %483, %498
  %504 = phi i64 [ %502, %498 ], [ 0, %483 ]
  %505 = shl i64 %487, 32
  %506 = ashr exact i64 %505, 32
  %507 = sub i64 %506, %19
  %508 = trunc i64 %507 to i32
  %509 = sdiv i32 %508, %16
  %510 = srem i32 %508, %16
  %511 = mul nsw i32 %509, %0
  %512 = add nsw i32 %511, %510
  %513 = icmp sgt i32 %512, -1
  %514 = icmp slt i32 %510, %0
  %515 = and i1 %514, %513
  %516 = icmp slt i32 %512, %17
  %517 = and i1 %516, %515
  br i1 %517, label %518, label %523

518:                                              ; preds = %503
  %519 = zext i32 %512 to i64
  %520 = getelementptr inbounds double, double* %6, i64 %519
  %521 = bitcast double* %520 to i64*
  %522 = load i64, i64* %521, align 8, !tbaa !7
  br label %523

523:                                              ; preds = %503, %518
  %524 = phi i64 [ %522, %518 ], [ 0, %503 ]
  %525 = shl i64 %507, 32
  %526 = ashr exact i64 %525, 32
  %527 = sub i64 %526, %19
  %528 = trunc i64 %527 to i32
  %529 = sdiv i32 %528, %16
  %530 = srem i32 %528, %16
  %531 = mul nsw i32 %529, %0
  %532 = add nsw i32 %531, %530
  %533 = icmp sgt i32 %532, -1
  %534 = icmp slt i32 %530, %0
  %535 = and i1 %534, %533
  %536 = icmp slt i32 %532, %17
  %537 = and i1 %536, %535
  br i1 %537, label %538, label %543

538:                                              ; preds = %523
  %539 = zext i32 %532 to i64
  %540 = getelementptr inbounds double, double* %6, i64 %539
  %541 = bitcast double* %540 to i64*
  %542 = load i64, i64* %541, align 8, !tbaa !7
  br label %543

543:                                              ; preds = %523, %538
  %544 = phi i64 [ %542, %538 ], [ 0, %523 ]
  %545 = shl i64 %527, 32
  %546 = ashr exact i64 %545, 32
  %547 = sub i64 %546, %19
  %548 = trunc i64 %547 to i32
  %549 = sdiv i32 %548, %16
  %550 = srem i32 %548, %16
  %551 = mul nsw i32 %549, %0
  %552 = add nsw i32 %551, %550
  %553 = icmp sgt i32 %552, -1
  %554 = icmp slt i32 %550, %0
  %555 = and i1 %554, %553
  %556 = icmp slt i32 %552, %17
  %557 = and i1 %556, %555
  br i1 %557, label %558, label %563

558:                                              ; preds = %543
  %559 = zext i32 %552 to i64
  %560 = getelementptr inbounds double, double* %6, i64 %559
  %561 = bitcast double* %560 to i64*
  %562 = load i64, i64* %561, align 8, !tbaa !7
  br label %563

563:                                              ; preds = %543, %558
  %564 = phi i64 [ %562, %558 ], [ 0, %543 ]
  %565 = shl i64 %547, 32
  %566 = ashr exact i64 %565, 32
  %567 = sub i64 %566, %19
  %568 = trunc i64 %567 to i32
  %569 = sdiv i32 %568, %16
  %570 = srem i32 %568, %16
  %571 = mul nsw i32 %569, %0
  %572 = add nsw i32 %571, %570
  %573 = icmp sgt i32 %572, -1
  %574 = icmp slt i32 %570, %0
  %575 = and i1 %574, %573
  %576 = icmp slt i32 %572, %17
  %577 = and i1 %576, %575
  br i1 %577, label %578, label %583

578:                                              ; preds = %563
  %579 = zext i32 %572 to i64
  %580 = getelementptr inbounds double, double* %6, i64 %579
  %581 = bitcast double* %580 to i64*
  %582 = load i64, i64* %581, align 8, !tbaa !7
  br label %583

583:                                              ; preds = %563, %578
  %584 = phi i64 [ %582, %578 ], [ 0, %563 ]
  %585 = shl i64 %567, 32
  %586 = ashr exact i64 %585, 32
  %587 = sub i64 %586, %19
  %588 = trunc i64 %587 to i32
  %589 = sdiv i32 %588, %16
  %590 = srem i32 %588, %16
  %591 = mul nsw i32 %589, %0
  %592 = add nsw i32 %591, %590
  %593 = icmp sgt i32 %592, -1
  %594 = icmp slt i32 %590, %0
  %595 = and i1 %594, %593
  %596 = icmp slt i32 %592, %17
  %597 = and i1 %596, %595
  br i1 %597, label %598, label %603

598:                                              ; preds = %583
  %599 = zext i32 %592 to i64
  %600 = getelementptr inbounds double, double* %6, i64 %599
  %601 = bitcast double* %600 to i64*
  %602 = load i64, i64* %601, align 8, !tbaa !7
  br label %603

603:                                              ; preds = %583, %598
  %604 = phi i64 [ %602, %598 ], [ 0, %583 ]
  %605 = shl i64 %587, 32
  %606 = ashr exact i64 %605, 32
  %607 = sub i64 %606, %19
  %608 = trunc i64 %607 to i32
  %609 = sdiv i32 %608, %16
  %610 = srem i32 %608, %16
  %611 = mul nsw i32 %609, %0
  %612 = add nsw i32 %611, %610
  %613 = icmp sgt i32 %612, -1
  %614 = icmp slt i32 %610, %0
  %615 = and i1 %614, %613
  %616 = icmp slt i32 %612, %17
  %617 = and i1 %616, %615
  br i1 %617, label %618, label %623

618:                                              ; preds = %603
  %619 = zext i32 %612 to i64
  %620 = getelementptr inbounds double, double* %6, i64 %619
  %621 = bitcast double* %620 to i64*
  %622 = load i64, i64* %621, align 8, !tbaa !7
  br label %623

623:                                              ; preds = %603, %618
  %624 = phi i64 [ %622, %618 ], [ 0, %603 ]
  %625 = shl i64 %607, 32
  %626 = ashr exact i64 %625, 32
  %627 = sub i64 %626, %19
  %628 = trunc i64 %627 to i32
  %629 = sdiv i32 %628, %16
  %630 = srem i32 %628, %16
  %631 = mul nsw i32 %629, %0
  %632 = add nsw i32 %631, %630
  %633 = icmp sgt i32 %632, -1
  %634 = icmp slt i32 %630, %0
  %635 = and i1 %634, %633
  %636 = icmp slt i32 %632, %17
  %637 = and i1 %636, %635
  br i1 %637, label %638, label %643

638:                                              ; preds = %623
  %639 = zext i32 %632 to i64
  %640 = getelementptr inbounds double, double* %6, i64 %639
  %641 = bitcast double* %640 to i64*
  %642 = load i64, i64* %641, align 8, !tbaa !7
  br label %643

643:                                              ; preds = %623, %638
  %644 = phi i64 [ %642, %638 ], [ 0, %623 ]
  %645 = shl i64 %627, 32
  %646 = ashr exact i64 %645, 32
  %647 = sub i64 %646, %19
  %648 = trunc i64 %647 to i32
  %649 = sdiv i32 %648, %16
  %650 = srem i32 %648, %16
  %651 = mul nsw i32 %649, %0
  %652 = add nsw i32 %651, %650
  %653 = icmp sgt i32 %652, -1
  %654 = icmp slt i32 %650, %0
  %655 = and i1 %654, %653
  %656 = icmp slt i32 %652, %17
  %657 = and i1 %656, %655
  br i1 %657, label %658, label %663

658:                                              ; preds = %643
  %659 = zext i32 %652 to i64
  %660 = getelementptr inbounds double, double* %6, i64 %659
  %661 = bitcast double* %660 to i64*
  %662 = load i64, i64* %661, align 8, !tbaa !7
  br label %663

663:                                              ; preds = %643, %658
  %664 = phi i64 [ %662, %658 ], [ 0, %643 ]
  %665 = shl i64 %647, 32
  %666 = ashr exact i64 %665, 32
  %667 = sub i64 %666, %19
  %668 = trunc i64 %667 to i32
  %669 = sdiv i32 %668, %16
  %670 = srem i32 %668, %16
  %671 = mul nsw i32 %669, %0
  %672 = add nsw i32 %671, %670
  %673 = icmp sgt i32 %672, -1
  %674 = icmp slt i32 %670, %0
  %675 = and i1 %674, %673
  %676 = icmp slt i32 %672, %17
  %677 = and i1 %676, %675
  br i1 %677, label %678, label %683

678:                                              ; preds = %663
  %679 = zext i32 %672 to i64
  %680 = getelementptr inbounds double, double* %6, i64 %679
  %681 = bitcast double* %680 to i64*
  %682 = load i64, i64* %681, align 8, !tbaa !7
  br label %683

683:                                              ; preds = %663, %678
  %684 = phi i64 [ %682, %678 ], [ 0, %663 ]
  %685 = sub i64 %667, %19
  %686 = trunc i64 %685 to i32
  %687 = sdiv i32 %686, %16
  %688 = srem i32 %686, %16
  %689 = mul nsw i32 %687, %0
  %690 = add nsw i32 %689, %688
  %691 = icmp sgt i32 %690, -1
  %692 = icmp slt i32 %688, %0
  %693 = and i1 %692, %691
  %694 = icmp slt i32 %690, %17
  %695 = and i1 %694, %693
  br i1 %695, label %696, label %701

696:                                              ; preds = %683
  %697 = zext i32 %690 to i64
  %698 = getelementptr inbounds double, double* %6, i64 %697
  %699 = bitcast double* %698 to i64*
  %700 = load i64, i64* %699, align 8, !tbaa !7
  br label %701

701:                                              ; preds = %683, %696
  %702 = phi i64 [ %700, %696 ], [ 0, %683 ]
  tail call void @_Z7barrierj(i32 1) #4
  br i1 %34, label %54, label %85

703:                                              ; preds = %93
  %704 = zext i32 %94 to i64
  %705 = getelementptr inbounds double, double* %5, i64 %704
  %706 = bitcast double* %705 to i64*
  store i64 %104, i64* %706, align 8, !tbaa !7
  br label %707

707:                                              ; preds = %703, %93
  %708 = sub i32 %94, %35
  %709 = icmp sgt i32 %708, -1
  %710 = icmp slt i32 %708, %17
  %711 = and i1 %709, %710
  br i1 %711, label %712, label %716

712:                                              ; preds = %707
  %713 = zext i32 %708 to i64
  %714 = getelementptr inbounds double, double* %5, i64 %713
  %715 = bitcast double* %714 to i64*
  store i64 %124, i64* %715, align 8, !tbaa !7
  br label %716

716:                                              ; preds = %712, %707
  %717 = sub i32 %708, %35
  %718 = icmp sgt i32 %717, -1
  %719 = icmp slt i32 %717, %17
  %720 = and i1 %718, %719
  br i1 %720, label %721, label %725

721:                                              ; preds = %716
  %722 = zext i32 %717 to i64
  %723 = getelementptr inbounds double, double* %5, i64 %722
  %724 = bitcast double* %723 to i64*
  store i64 %144, i64* %724, align 8, !tbaa !7
  br label %725

725:                                              ; preds = %721, %716
  %726 = sub i32 %717, %35
  %727 = icmp sgt i32 %726, -1
  %728 = icmp slt i32 %726, %17
  %729 = and i1 %727, %728
  br i1 %729, label %730, label %734

730:                                              ; preds = %725
  %731 = zext i32 %726 to i64
  %732 = getelementptr inbounds double, double* %5, i64 %731
  %733 = bitcast double* %732 to i64*
  store i64 %164, i64* %733, align 8, !tbaa !7
  br label %734

734:                                              ; preds = %730, %725
  %735 = sub i32 %726, %35
  %736 = icmp sgt i32 %735, -1
  %737 = icmp slt i32 %735, %17
  %738 = and i1 %736, %737
  br i1 %738, label %739, label %743

739:                                              ; preds = %734
  %740 = zext i32 %735 to i64
  %741 = getelementptr inbounds double, double* %5, i64 %740
  %742 = bitcast double* %741 to i64*
  store i64 %184, i64* %742, align 8, !tbaa !7
  br label %743

743:                                              ; preds = %739, %734
  %744 = sub i32 %735, %35
  %745 = icmp sgt i32 %744, -1
  %746 = icmp slt i32 %744, %17
  %747 = and i1 %745, %746
  br i1 %747, label %748, label %752

748:                                              ; preds = %743
  %749 = zext i32 %744 to i64
  %750 = getelementptr inbounds double, double* %5, i64 %749
  %751 = bitcast double* %750 to i64*
  store i64 %204, i64* %751, align 8, !tbaa !7
  br label %752

752:                                              ; preds = %748, %743
  %753 = sub i32 %744, %35
  %754 = icmp sgt i32 %753, -1
  %755 = icmp slt i32 %753, %17
  %756 = and i1 %754, %755
  br i1 %756, label %757, label %761

757:                                              ; preds = %752
  %758 = zext i32 %753 to i64
  %759 = getelementptr inbounds double, double* %5, i64 %758
  %760 = bitcast double* %759 to i64*
  store i64 %224, i64* %760, align 8, !tbaa !7
  br label %761

761:                                              ; preds = %757, %752
  %762 = sub i32 %753, %35
  %763 = icmp sgt i32 %762, -1
  %764 = icmp slt i32 %762, %17
  %765 = and i1 %763, %764
  br i1 %765, label %766, label %770

766:                                              ; preds = %761
  %767 = zext i32 %762 to i64
  %768 = getelementptr inbounds double, double* %5, i64 %767
  %769 = bitcast double* %768 to i64*
  store i64 %244, i64* %769, align 8, !tbaa !7
  br label %770

770:                                              ; preds = %766, %761
  %771 = sub i32 %762, %35
  %772 = icmp sgt i32 %771, -1
  %773 = icmp slt i32 %771, %17
  %774 = and i1 %772, %773
  br i1 %774, label %775, label %779

775:                                              ; preds = %770
  %776 = zext i32 %771 to i64
  %777 = getelementptr inbounds double, double* %5, i64 %776
  %778 = bitcast double* %777 to i64*
  store i64 %264, i64* %778, align 8, !tbaa !7
  br label %779

779:                                              ; preds = %775, %770
  %780 = sub i32 %771, %35
  %781 = icmp sgt i32 %780, -1
  %782 = icmp slt i32 %780, %17
  %783 = and i1 %781, %782
  br i1 %783, label %784, label %788

784:                                              ; preds = %779
  %785 = zext i32 %780 to i64
  %786 = getelementptr inbounds double, double* %5, i64 %785
  %787 = bitcast double* %786 to i64*
  store i64 %284, i64* %787, align 8, !tbaa !7
  br label %788

788:                                              ; preds = %784, %779
  %789 = sub i32 %780, %35
  %790 = icmp sgt i32 %789, -1
  %791 = icmp slt i32 %789, %17
  %792 = and i1 %790, %791
  br i1 %792, label %793, label %797

793:                                              ; preds = %788
  %794 = zext i32 %789 to i64
  %795 = getelementptr inbounds double, double* %5, i64 %794
  %796 = bitcast double* %795 to i64*
  store i64 %304, i64* %796, align 8, !tbaa !7
  br label %797

797:                                              ; preds = %793, %788
  %798 = sub i32 %789, %35
  %799 = icmp sgt i32 %798, -1
  %800 = icmp slt i32 %798, %17
  %801 = and i1 %799, %800
  br i1 %801, label %802, label %806

802:                                              ; preds = %797
  %803 = zext i32 %798 to i64
  %804 = getelementptr inbounds double, double* %5, i64 %803
  %805 = bitcast double* %804 to i64*
  store i64 %324, i64* %805, align 8, !tbaa !7
  br label %806

806:                                              ; preds = %802, %797
  %807 = sub i32 %798, %35
  %808 = icmp sgt i32 %807, -1
  %809 = icmp slt i32 %807, %17
  %810 = and i1 %808, %809
  br i1 %810, label %811, label %815

811:                                              ; preds = %806
  %812 = zext i32 %807 to i64
  %813 = getelementptr inbounds double, double* %5, i64 %812
  %814 = bitcast double* %813 to i64*
  store i64 %344, i64* %814, align 8, !tbaa !7
  br label %815

815:                                              ; preds = %811, %806
  %816 = sub i32 %807, %35
  %817 = icmp sgt i32 %816, -1
  %818 = icmp slt i32 %816, %17
  %819 = and i1 %817, %818
  br i1 %819, label %820, label %824

820:                                              ; preds = %815
  %821 = zext i32 %816 to i64
  %822 = getelementptr inbounds double, double* %5, i64 %821
  %823 = bitcast double* %822 to i64*
  store i64 %364, i64* %823, align 8, !tbaa !7
  br label %824

824:                                              ; preds = %820, %815
  %825 = sub i32 %816, %35
  %826 = icmp sgt i32 %825, -1
  %827 = icmp slt i32 %825, %17
  %828 = and i1 %826, %827
  br i1 %828, label %829, label %833

829:                                              ; preds = %824
  %830 = zext i32 %825 to i64
  %831 = getelementptr inbounds double, double* %5, i64 %830
  %832 = bitcast double* %831 to i64*
  store i64 %384, i64* %832, align 8, !tbaa !7
  br label %833

833:                                              ; preds = %829, %824
  %834 = sub i32 %825, %35
  %835 = icmp sgt i32 %834, -1
  %836 = icmp slt i32 %834, %17
  %837 = and i1 %835, %836
  br i1 %837, label %838, label %842

838:                                              ; preds = %833
  %839 = zext i32 %834 to i64
  %840 = getelementptr inbounds double, double* %5, i64 %839
  %841 = bitcast double* %840 to i64*
  store i64 %404, i64* %841, align 8, !tbaa !7
  br label %842

842:                                              ; preds = %838, %833
  %843 = sub i32 %834, %35
  %844 = icmp sgt i32 %843, -1
  %845 = icmp slt i32 %843, %17
  %846 = and i1 %844, %845
  br i1 %846, label %847, label %851

847:                                              ; preds = %842
  %848 = zext i32 %843 to i64
  %849 = getelementptr inbounds double, double* %5, i64 %848
  %850 = bitcast double* %849 to i64*
  store i64 %424, i64* %850, align 8, !tbaa !7
  br label %851

851:                                              ; preds = %847, %842
  %852 = sub i32 %843, %35
  %853 = icmp sgt i32 %852, -1
  %854 = icmp slt i32 %852, %17
  %855 = and i1 %853, %854
  br i1 %855, label %856, label %860

856:                                              ; preds = %851
  %857 = zext i32 %852 to i64
  %858 = getelementptr inbounds double, double* %5, i64 %857
  %859 = bitcast double* %858 to i64*
  store i64 %444, i64* %859, align 8, !tbaa !7
  br label %860

860:                                              ; preds = %856, %851
  %861 = sub i32 %852, %35
  %862 = icmp sgt i32 %861, -1
  %863 = icmp slt i32 %861, %17
  %864 = and i1 %862, %863
  br i1 %864, label %865, label %869

865:                                              ; preds = %860
  %866 = zext i32 %861 to i64
  %867 = getelementptr inbounds double, double* %5, i64 %866
  %868 = bitcast double* %867 to i64*
  store i64 %464, i64* %868, align 8, !tbaa !7
  br label %869

869:                                              ; preds = %865, %860
  %870 = sub i32 %861, %35
  %871 = icmp sgt i32 %870, -1
  %872 = icmp slt i32 %870, %17
  %873 = and i1 %871, %872
  br i1 %873, label %874, label %878

874:                                              ; preds = %869
  %875 = zext i32 %870 to i64
  %876 = getelementptr inbounds double, double* %5, i64 %875
  %877 = bitcast double* %876 to i64*
  store i64 %484, i64* %877, align 8, !tbaa !7
  br label %878

878:                                              ; preds = %874, %869
  %879 = sub i32 %870, %35
  %880 = icmp sgt i32 %879, -1
  %881 = icmp slt i32 %879, %17
  %882 = and i1 %880, %881
  br i1 %882, label %883, label %887

883:                                              ; preds = %878
  %884 = zext i32 %879 to i64
  %885 = getelementptr inbounds double, double* %5, i64 %884
  %886 = bitcast double* %885 to i64*
  store i64 %504, i64* %886, align 8, !tbaa !7
  br label %887

887:                                              ; preds = %883, %878
  %888 = sub i32 %879, %35
  %889 = icmp sgt i32 %888, -1
  %890 = icmp slt i32 %888, %17
  %891 = and i1 %889, %890
  br i1 %891, label %892, label %896

892:                                              ; preds = %887
  %893 = zext i32 %888 to i64
  %894 = getelementptr inbounds double, double* %5, i64 %893
  %895 = bitcast double* %894 to i64*
  store i64 %524, i64* %895, align 8, !tbaa !7
  br label %896

896:                                              ; preds = %892, %887
  %897 = sub i32 %888, %35
  %898 = icmp sgt i32 %897, -1
  %899 = icmp slt i32 %897, %17
  %900 = and i1 %898, %899
  br i1 %900, label %901, label %905

901:                                              ; preds = %896
  %902 = zext i32 %897 to i64
  %903 = getelementptr inbounds double, double* %5, i64 %902
  %904 = bitcast double* %903 to i64*
  store i64 %544, i64* %904, align 8, !tbaa !7
  br label %905

905:                                              ; preds = %901, %896
  %906 = sub i32 %897, %35
  %907 = icmp sgt i32 %906, -1
  %908 = icmp slt i32 %906, %17
  %909 = and i1 %907, %908
  br i1 %909, label %910, label %914

910:                                              ; preds = %905
  %911 = zext i32 %906 to i64
  %912 = getelementptr inbounds double, double* %5, i64 %911
  %913 = bitcast double* %912 to i64*
  store i64 %564, i64* %913, align 8, !tbaa !7
  br label %914

914:                                              ; preds = %910, %905
  %915 = sub i32 %906, %35
  %916 = icmp sgt i32 %915, -1
  %917 = icmp slt i32 %915, %17
  %918 = and i1 %916, %917
  br i1 %918, label %919, label %923

919:                                              ; preds = %914
  %920 = zext i32 %915 to i64
  %921 = getelementptr inbounds double, double* %5, i64 %920
  %922 = bitcast double* %921 to i64*
  store i64 %584, i64* %922, align 8, !tbaa !7
  br label %923

923:                                              ; preds = %919, %914
  %924 = sub i32 %915, %35
  %925 = icmp sgt i32 %924, -1
  %926 = icmp slt i32 %924, %17
  %927 = and i1 %925, %926
  br i1 %927, label %928, label %932

928:                                              ; preds = %923
  %929 = zext i32 %924 to i64
  %930 = getelementptr inbounds double, double* %5, i64 %929
  %931 = bitcast double* %930 to i64*
  store i64 %604, i64* %931, align 8, !tbaa !7
  br label %932

932:                                              ; preds = %928, %923
  %933 = sub i32 %924, %35
  %934 = icmp sgt i32 %933, -1
  %935 = icmp slt i32 %933, %17
  %936 = and i1 %934, %935
  br i1 %936, label %937, label %941

937:                                              ; preds = %932
  %938 = zext i32 %933 to i64
  %939 = getelementptr inbounds double, double* %5, i64 %938
  %940 = bitcast double* %939 to i64*
  store i64 %624, i64* %940, align 8, !tbaa !7
  br label %941

941:                                              ; preds = %937, %932
  %942 = sub i32 %933, %35
  %943 = icmp sgt i32 %942, -1
  %944 = icmp slt i32 %942, %17
  %945 = and i1 %943, %944
  br i1 %945, label %946, label %950

946:                                              ; preds = %941
  %947 = zext i32 %942 to i64
  %948 = getelementptr inbounds double, double* %5, i64 %947
  %949 = bitcast double* %948 to i64*
  store i64 %644, i64* %949, align 8, !tbaa !7
  br label %950

950:                                              ; preds = %946, %941
  %951 = sub i32 %942, %35
  %952 = icmp sgt i32 %951, -1
  %953 = icmp slt i32 %951, %17
  %954 = and i1 %952, %953
  br i1 %954, label %955, label %959

955:                                              ; preds = %950
  %956 = zext i32 %951 to i64
  %957 = getelementptr inbounds double, double* %5, i64 %956
  %958 = bitcast double* %957 to i64*
  store i64 %664, i64* %958, align 8, !tbaa !7
  br label %959

959:                                              ; preds = %955, %950
  %960 = sub i32 %951, %35
  %961 = icmp sgt i32 %960, -1
  %962 = icmp slt i32 %960, %17
  %963 = and i1 %961, %962
  br i1 %963, label %964, label %968

964:                                              ; preds = %959
  %965 = zext i32 %960 to i64
  %966 = getelementptr inbounds double, double* %5, i64 %965
  %967 = bitcast double* %966 to i64*
  store i64 %684, i64* %967, align 8, !tbaa !7
  br label %968

968:                                              ; preds = %964, %959
  %969 = sub i32 %960, %35
  %970 = icmp sgt i32 %969, -1
  %971 = icmp slt i32 %969, %17
  %972 = and i1 %970, %971
  br i1 %972, label %973, label %977

973:                                              ; preds = %968
  %974 = zext i32 %969 to i64
  %975 = getelementptr inbounds double, double* %5, i64 %974
  %976 = bitcast double* %975 to i64*
  store i64 %702, i64* %976, align 8, !tbaa !7
  br label %977

977:                                              ; preds = %973, %968
  %978 = tail call i64 @_Z14get_num_groupsj(i32 0) #3
  %979 = trunc i64 %978 to i32
  %980 = add i32 %38, %979
  %981 = icmp slt i32 %980, %3
  br i1 %981, label %37, label %36
}

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_local_sizej(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_local_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local void @_Z7barrierj(i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local i32 @_Z10atomic_addPU8CLglobalVii(i32*, i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_group_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_num_groupsj(i32) local_unnamed_addr #1

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
!3 = !{i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1}
!4 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!5 = !{!"int", !"int", !"int", !"int", !"float", !"double*", !"double*", !"int*"}
!6 = !{!"", !"", !"", !"", !"", !"", !"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"double", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
