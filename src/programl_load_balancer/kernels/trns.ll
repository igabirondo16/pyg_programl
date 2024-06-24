; ModuleID = './kernels/trns.cl'
source_filename = "./kernels/trns.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: convergent nounwind uwtable
define dso_local spir_kernel void @PTTWAC_soa_asta(i32 %0, i32 %1, i32 %2, i32* nocapture %3, i32* nocapture %4, double* nocapture %5, i32* %6, i32* %7) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %9 = tail call i64 @_Z12get_local_idj(i32 0) #3
  %10 = trunc i64 %9 to i32
  %11 = mul nsw i32 %1, %0
  %12 = add nsw i32 %11, -1
  %13 = icmp eq i32 %10, 0
  br i1 %13, label %14, label %16

14:                                               ; preds = %8
  %15 = tail call i32 @_Z8atom_addPU8CLglobalVii(i32* %7, i32 1) #4
  store i32 %15, i32* %4, align 4, !tbaa !7
  br label %16

16:                                               ; preds = %14, %8
  tail call void @_Z7barrierj(i32 1) #4
  %17 = load i32, i32* %4, align 4, !tbaa !7
  %18 = icmp slt i32 %17, %12
  br i1 %18, label %19, label %193

19:                                               ; preds = %16
  %20 = icmp slt i32 %10, %2
  br label %21

21:                                               ; preds = %19, %182
  %22 = phi i32 [ %17, %19 ], [ %191, %182 ]
  %23 = phi double [ undef, %19 ], [ %190, %182 ]
  %24 = phi double [ undef, %19 ], [ %189, %182 ]
  %25 = phi double [ undef, %19 ], [ %188, %182 ]
  %26 = phi double [ undef, %19 ], [ %187, %182 ]
  %27 = phi double [ undef, %19 ], [ %186, %182 ]
  %28 = phi double [ undef, %19 ], [ %185, %182 ]
  %29 = phi double [ undef, %19 ], [ %184, %182 ]
  %30 = phi double [ undef, %19 ], [ %183, %182 ]
  %31 = mul nsw i32 %22, %0
  %32 = sdiv i32 %22, %1
  %33 = mul nsw i32 %32, %12
  %34 = sub nsw i32 %31, %33
  %35 = icmp eq i32 %34, %22
  br i1 %35, label %36, label %39

36:                                               ; preds = %21
  br i1 %13, label %37, label %182

37:                                               ; preds = %36
  %38 = tail call i32 @_Z8atom_addPU8CLglobalVii(i32* %7, i32 1) #4
  store i32 %38, i32* %4, align 4, !tbaa !7
  br label %182

39:                                               ; preds = %21
  br i1 %20, label %40, label %46

40:                                               ; preds = %39
  %41 = mul nsw i32 %22, %2
  %42 = add nsw i32 %41, %10
  %43 = sext i32 %42 to i64
  %44 = getelementptr inbounds double, double* %5, i64 %43
  %45 = load double, double* %44, align 8, !tbaa !11
  br label %46

46:                                               ; preds = %40, %39
  %47 = phi double [ %45, %40 ], [ %30, %39 ]
  %48 = tail call i64 @_Z14get_local_sizej(i32 0) #3
  %49 = add i64 %48, %9
  %50 = trunc i64 %49 to i32
  %51 = icmp slt i32 %50, %2
  br i1 %51, label %52, label %58

52:                                               ; preds = %46
  %53 = mul nsw i32 %22, %2
  %54 = add nsw i32 %53, %50
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds double, double* %5, i64 %55
  %57 = load double, double* %56, align 8, !tbaa !11
  br label %58

58:                                               ; preds = %52, %46
  %59 = phi double [ %57, %52 ], [ %29, %46 ]
  %60 = add i64 %49, %48
  %61 = trunc i64 %60 to i32
  %62 = icmp slt i32 %61, %2
  br i1 %62, label %63, label %69

63:                                               ; preds = %58
  %64 = mul nsw i32 %22, %2
  %65 = add nsw i32 %64, %61
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds double, double* %5, i64 %66
  %68 = load double, double* %67, align 8, !tbaa !11
  br label %69

69:                                               ; preds = %63, %58
  %70 = phi double [ %68, %63 ], [ %28, %58 ]
  %71 = add i64 %60, %48
  %72 = trunc i64 %71 to i32
  %73 = icmp slt i32 %72, %2
  br i1 %73, label %74, label %80

74:                                               ; preds = %69
  %75 = mul nsw i32 %22, %2
  %76 = add nsw i32 %75, %72
  %77 = sext i32 %76 to i64
  %78 = getelementptr inbounds double, double* %5, i64 %77
  %79 = load double, double* %78, align 8, !tbaa !11
  br label %80

80:                                               ; preds = %74, %69
  %81 = phi double [ %79, %74 ], [ %27, %69 ]
  br i1 %13, label %82, label %86

82:                                               ; preds = %80
  %83 = sext i32 %22 to i64
  %84 = getelementptr inbounds i32, i32* %6, i64 %83
  %85 = tail call i32 @_Z8atom_addPU8CLglobalVii(i32* %84, i32 0) #4
  store i32 %85, i32* %3, align 4, !tbaa !7
  br label %86

86:                                               ; preds = %82, %80
  tail call void @_Z7barrierj(i32 1) #4
  %87 = load i32, i32* %3, align 4, !tbaa !7
  %88 = icmp eq i32 %87, 0
  br i1 %88, label %89, label %171

89:                                               ; preds = %86, %162
  %90 = phi double [ %130, %162 ], [ %23, %86 ]
  %91 = phi double [ %122, %162 ], [ %24, %86 ]
  %92 = phi double [ %114, %162 ], [ %25, %86 ]
  %93 = phi double [ %106, %162 ], [ %26, %86 ]
  %94 = phi double [ %166, %162 ], [ %81, %86 ]
  %95 = phi double [ %165, %162 ], [ %70, %86 ]
  %96 = phi double [ %164, %162 ], [ %59, %86 ]
  %97 = phi double [ %163, %162 ], [ %47, %86 ]
  %98 = phi i32 [ %170, %162 ], [ %34, %86 ]
  br i1 %20, label %99, label %105

99:                                               ; preds = %89
  %100 = mul nsw i32 %98, %2
  %101 = add nsw i32 %100, %10
  %102 = sext i32 %101 to i64
  %103 = getelementptr inbounds double, double* %5, i64 %102
  %104 = load double, double* %103, align 8, !tbaa !11
  br label %105

105:                                              ; preds = %99, %89
  %106 = phi double [ %104, %99 ], [ %93, %89 ]
  br i1 %51, label %107, label %113

107:                                              ; preds = %105
  %108 = mul nsw i32 %98, %2
  %109 = add nsw i32 %108, %50
  %110 = sext i32 %109 to i64
  %111 = getelementptr inbounds double, double* %5, i64 %110
  %112 = load double, double* %111, align 8, !tbaa !11
  br label %113

113:                                              ; preds = %107, %105
  %114 = phi double [ %112, %107 ], [ %92, %105 ]
  br i1 %62, label %115, label %121

115:                                              ; preds = %113
  %116 = mul nsw i32 %98, %2
  %117 = add nsw i32 %116, %61
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds double, double* %5, i64 %118
  %120 = load double, double* %119, align 8, !tbaa !11
  br label %121

121:                                              ; preds = %115, %113
  %122 = phi double [ %120, %115 ], [ %91, %113 ]
  br i1 %73, label %123, label %129

123:                                              ; preds = %121
  %124 = mul nsw i32 %98, %2
  %125 = add nsw i32 %124, %72
  %126 = sext i32 %125 to i64
  %127 = getelementptr inbounds double, double* %5, i64 %126
  %128 = load double, double* %127, align 8, !tbaa !11
  br label %129

129:                                              ; preds = %123, %121
  %130 = phi double [ %128, %123 ], [ %90, %121 ]
  br i1 %13, label %131, label %135

131:                                              ; preds = %129
  %132 = sext i32 %98 to i64
  %133 = getelementptr inbounds i32, i32* %6, i64 %132
  %134 = tail call i32 @_Z11atomic_xchgPU8CLglobalVii(i32* %133, i32 1) #4
  store i32 %134, i32* %3, align 4, !tbaa !7
  br label %135

135:                                              ; preds = %131, %129
  tail call void @_Z7barrierj(i32 1) #4
  %136 = load i32, i32* %3, align 4, !tbaa !7
  %137 = icmp eq i32 %136, 0
  br i1 %137, label %138, label %162

138:                                              ; preds = %135
  br i1 %20, label %139, label %144

139:                                              ; preds = %138
  %140 = mul nsw i32 %98, %2
  %141 = add nsw i32 %140, %10
  %142 = sext i32 %141 to i64
  %143 = getelementptr inbounds double, double* %5, i64 %142
  store double %97, double* %143, align 8, !tbaa !11
  br label %144

144:                                              ; preds = %139, %138
  br i1 %51, label %145, label %150

145:                                              ; preds = %144
  %146 = mul nsw i32 %98, %2
  %147 = add nsw i32 %146, %50
  %148 = sext i32 %147 to i64
  %149 = getelementptr inbounds double, double* %5, i64 %148
  store double %96, double* %149, align 8, !tbaa !11
  br label %150

150:                                              ; preds = %145, %144
  br i1 %62, label %151, label %156

151:                                              ; preds = %150
  %152 = mul nsw i32 %98, %2
  %153 = add nsw i32 %152, %61
  %154 = sext i32 %153 to i64
  %155 = getelementptr inbounds double, double* %5, i64 %154
  store double %95, double* %155, align 8, !tbaa !11
  br label %156

156:                                              ; preds = %151, %150
  br i1 %73, label %157, label %162

157:                                              ; preds = %156
  %158 = mul nsw i32 %98, %2
  %159 = add nsw i32 %158, %72
  %160 = sext i32 %159 to i64
  %161 = getelementptr inbounds double, double* %5, i64 %160
  store double %94, double* %161, align 8, !tbaa !11
  br label %162

162:                                              ; preds = %135, %156, %157
  %163 = select i1 %20, double %106, double %97
  %164 = select i1 %51, double %114, double %96
  %165 = select i1 %62, double %122, double %95
  %166 = select i1 %73, double %130, double %94
  %167 = mul nsw i32 %98, %0
  %168 = sdiv i32 %98, %1
  %169 = mul nsw i32 %168, %12
  %170 = sub nsw i32 %167, %169
  br i1 %137, label %89, label %171

171:                                              ; preds = %162, %86
  %172 = phi double [ %47, %86 ], [ %163, %162 ]
  %173 = phi double [ %59, %86 ], [ %164, %162 ]
  %174 = phi double [ %70, %86 ], [ %165, %162 ]
  %175 = phi double [ %81, %86 ], [ %166, %162 ]
  %176 = phi double [ %26, %86 ], [ %106, %162 ]
  %177 = phi double [ %25, %86 ], [ %114, %162 ]
  %178 = phi double [ %24, %86 ], [ %122, %162 ]
  %179 = phi double [ %23, %86 ], [ %130, %162 ]
  br i1 %13, label %180, label %182

180:                                              ; preds = %171
  %181 = tail call i32 @_Z8atom_addPU8CLglobalVii(i32* %7, i32 1) #4
  store i32 %181, i32* %4, align 4, !tbaa !7
  br label %182

182:                                              ; preds = %171, %180, %36, %37
  %183 = phi double [ %30, %37 ], [ %30, %36 ], [ %172, %180 ], [ %172, %171 ]
  %184 = phi double [ %29, %37 ], [ %29, %36 ], [ %173, %180 ], [ %173, %171 ]
  %185 = phi double [ %28, %37 ], [ %28, %36 ], [ %174, %180 ], [ %174, %171 ]
  %186 = phi double [ %27, %37 ], [ %27, %36 ], [ %175, %180 ], [ %175, %171 ]
  %187 = phi double [ %26, %37 ], [ %26, %36 ], [ %176, %180 ], [ %176, %171 ]
  %188 = phi double [ %25, %37 ], [ %25, %36 ], [ %177, %180 ], [ %177, %171 ]
  %189 = phi double [ %24, %37 ], [ %24, %36 ], [ %178, %180 ], [ %178, %171 ]
  %190 = phi double [ %23, %37 ], [ %23, %36 ], [ %179, %180 ], [ %179, %171 ]
  tail call void @_Z7barrierj(i32 1) #4
  %191 = load i32, i32* %4, align 4, !tbaa !7
  %192 = icmp slt i32 %191, %12
  br i1 %192, label %21, label %193

193:                                              ; preds = %182, %16
  ret void
}

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z12get_local_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local i32 @_Z8atom_addPU8CLglobalVii(i32*, i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local void @_Z7barrierj(i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z14get_local_sizej(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local i32 @_Z11atomic_xchgPU8CLglobalVii(i32*, i32) local_unnamed_addr #2

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
!3 = !{i32 0, i32 0, i32 0, i32 3, i32 3, i32 1, i32 1, i32 1}
!4 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!5 = !{!"int", !"int", !"int", !"int*", !"int*", !"double*", !"int*", !"int*"}
!6 = !{!"", !"", !"", !"", !"", !"", !"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !9, i64 0}
