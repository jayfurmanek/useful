#define EIGEN_USE_GPU
#define EIGEN_USE_HIP
#include "unsupported/Eigen/CXX11/Tensor"
#include <stdio.h>
#include <vector>
#include <hip/hip_runtime.h>

using ScalarType = Eigen::half;
using IndexType = int;
constexpr auto DataLayout = Eigen::RowMajor;
namespace Eigen {
	namespace internal {
		template <typename T>
		struct AvgPoolMeanReducer {
			static const bool PacketAccess = false;
			static constexpr bool IsStateful = true;
			EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE AvgPoolMeanReducer() : scalarCount_(0) {
				typedef typename packet_traits<T>::type Packet;
				//packetCount_ = pset1<Packet>(T(0.0));
				packetCount_ = 0;
			}
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) {
				if (t != -Eigen::NumTraits<T>::highest()) {
					(*accum) = (*accum) + t;
					scalarCount_++;
				}
			}
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {return static_cast<T>(0);}
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
				eigen_assert(scalarCount_ > 0);
				return accum / T(scalarCount_);
			}
			
			protected:
			//typedef typename packet_traits<T>::type Packet;
			int scalarCount_;
			//Packet packetCount_;
			int packetCount_;
		};
	}// namespace internal
	
	template <typename Input>
	EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
		const Eigen::DSizes<typename internal::traits<Input>::Index,
			internal::traits<Input>::NumDimensions>,
			const TensorReductionOp<
				internal::AvgPoolMeanReducer<typename internal::remove_const<
					typename internal::traits<Input>::Scalar>::type>,
					typename internal::conditional<internal::traits<Input>::Layout == ColMajor,
					const Eigen::IndexList<Eigen::type2index<1>,
					Eigen::type2index<2>
				>,
				const Eigen::IndexList<Eigen::type2index<2>,
				Eigen::type2index<3>
			>
		>::type,
			const TensorImagePatchOp<Dynamic,
			Dynamic, const Input>
		>
	>
	SpatialAvgPooling(const Input& input,
	DenseIndex patchRows,
	DenseIndex patchCols,
	DenseIndex strideRows,
	DenseIndex strideCols,
	const PaddingType padding_type,
	DenseIndex in_strideRows = 1,
	DenseIndex in_strideCols = 1) 
	{
		EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 4,YOU_MADE_A_PROGRAMMING_MISTAKE);
	
		typedef typename internal::traits<Input>::Index TensorIndex;
		TensorRef<
			Tensor<
				typename internal::traits<Input>::Scalar,internal::traits<Input>::NumDimensions,
				internal::traits<Input>::Layout,
				TensorIndex
				>
			>in(input);
		const DenseIndex patchRowsEff = patchRows + (patchRows - 1) * (in_strideRows - 1);
		const DenseIndex patchColsEff = patchCols + (patchCols - 1) * (in_strideCols - 1);
		static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
		static const int idxRows = isColMajor ? 1 : 2;
		static const int idxCols = isColMajor ? 2 : 1;
		// Molds the output of the reduction into the shape expected by the user.
		// (assuming col-major):
		// - 1st dim: channels
		// - 2nd dim: output height
		// - 3rd dim: output width
		// - 4th dim and beyond: everything else including batch size

		Eigen::DSizes<TensorIndex, internal::traits<Input>::NumDimensions>post_reduce_dims;post_reduce_dims[0] = in.dimension(0);
	
		if (padding_type == PADDING_VALID) {
			post_reduce_dims[idxRows] = Eigen::divup(
				static_cast<DenseIndex>(in.dimension(idxRows)) - patchRowsEff + 1,strideRows);
				post_reduce_dims[idxCols] = Eigen::divup(static_cast<DenseIndex>(in.dimension(idxCols)) - patchColsEff + 1,strideCols);
		}
		else {
			post_reduce_dims[idxRows] = Eigen::divup(static_cast<DenseIndex>(in.dimension(idxRows)), strideRows);
			post_reduce_dims[idxCols] = Eigen::divup(static_cast<DenseIndex>(in.dimension(idxCols)), strideCols);
		}
		post_reduce_dims[3] = in.dimension(3);
	
		typedef typename internal::remove_const<typename internal::traits<Input>::Scalar>::type CoeffReturnType;
		internal::AvgPoolMeanReducer<CoeffReturnType> mean_with_nan;
	
		// Take advantage of cxx11 to give the compiler information it can use to
		// optimize the code.
		typename internal::conditional<
			internal::traits<Input>::Layout == ColMajor,
			const Eigen::IndexList<Eigen::type2index<1>,
				Eigen::type2index<2>
			>,
			const Eigen::IndexList<
				Eigen::type2index<2>,
				Eigen::type2index<3>
			>
		>::type reduction_dims;
		return input.extract_image_patches(patchRows, patchCols, strideRows, strideCols, in_strideRows, in_strideCols, padding_type,-Eigen::NumTraits<CoeffReturnType>::highest()).reduce(reduction_dims, mean_with_nan).reshape(post_reduce_dims);
	}
}

int main(){
	Eigen::GpuStreamDevice stream;
	Eigen::GpuDevice d(&stream);
	
	IndexType input_size = 49;
	int input_bytes = input_size * sizeof(ScalarType);
	std::vector<ScalarType> input_host;
	input_host.resize(input_size);
	
	for(int i=0; i<input_size; i++)
		input_host[i] = ScalarType(float(i + 1));

	ScalarType* input_device;
	hipMalloc((void**)(&input_device), input_bytes);
	hipMemcpy(input_device, &input_host[0], input_bytes, hipMemcpyHostToDevice);
	
	IndexType output_size = 4;
	int output_bytes = output_size * sizeof(ScalarType);
	ScalarType* output_device;
	
	hipMalloc((void**)(&output_device), output_bytes);
	auto input_tensor = Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>(input_device, 1, 7, 7, 1);
	auto output_tensor = Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>(output_device, 1, 2, 2, 1);
		
	output_tensor.device(d) = SpatialAvgPooling(input_tensor, 2, 2, 3, 3, Eigen::PaddingType::PADDING_VALID);
	std::vector<ScalarType> output_host;output_host.resize(output_size);
	
	hipMemcpy(&output_host[0], output_device, output_bytes, hipMemcpyDeviceToHost);
	
	for(int i=0; i<output_size; i++)
		printf("%f\n", float(output_host[i]));
}
