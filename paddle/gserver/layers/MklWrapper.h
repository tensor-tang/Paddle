/*******************************************************************************
* Copyright 2015-2016 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

#pragma once

#include <stdarg.h>
#include <stddef.h>

#include "mkl_dnn_types.h"
#include "mkl_dnn.h"
//#include "paddle/utils/TypeDefs.h"

namespace paddle {

#ifdef PADDLE_TYPE_DOUBLE
/*******************************************************************************
 * F64 section: double precision 
 ******************************************************************************/
dnnError_t dnnLayoutCreate(
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]) {
    return dnnLayoutCreate_F64(pLayout, dimension, size, strides);
}

dnnError_t dnnLayoutCreateFromPrimitive(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type) {
    return dnnLayoutCreateFromPrimitive_F64(pLayout, primitive, type);
}

size_t dnnLayoutGetMemorySize(
        const dnnLayout_t layout) {
    return dnnLayoutGetMemorySize_F64(layout);
}

int dnnLayoutCompare(
        const dnnLayout_t l1, const dnnLayout_t l2) {
    return dnnLayoutCompare_F64(l1, l2);
}

dnnError_t dnnAllocateBuffer(
        void **pPtr, dnnLayout_t layout) {
    return dnnAllocateBuffer_F64(pPtr, layout);
}

dnnError_t dnnReleaseBuffer(
        void *ptr) {
    return dnnReleaseBuffer_F64(ptr);
}

dnnError_t dnnLayoutDelete(
        dnnLayout_t layout) {
    return dnnLayoutDelete_F64(layout);
}

dnnError_t dnnPrimitiveAttributesCreate(
        dnnPrimitiveAttributes_t *attributes) {
    return dnnPrimitiveAttributesCreate_F64(attributes);
}

dnnError_t dnnPrimitiveAttributesDestroy(
        dnnPrimitiveAttributes_t attributes) {
    return dnnPrimitiveAttributesDestroy_F64(attributes);
}

dnnError_t dnnPrimitiveGetAttributes(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes) {
    return dnnPrimitiveGetAttributes_F64(primitive, attributes);
}

dnnError_t dnnExecute(
        dnnPrimitive_t primitive, void *resources[]) {
    return dnnExecute_F64(primitive, resources);
}

dnnError_t dnnExecuteAsync(
        dnnPrimitive_t primitive, void *resources[]) {
    return dnnExecuteAsync_F64(primitive, resources);
}

dnnError_t dnnWaitFor(
        dnnPrimitive_t primitive) {
    return dnnWaitFor_F64(primitive);
}

dnnError_t dnnDelete(
        dnnPrimitive_t primitive) {
    return dnnDelete_F64(primitive);
}

dnnError_t dnnConversionCreate(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to) {
    return dnnConversionCreate_F64(pConversion, from, to);
}

dnnError_t dnnConversionExecute(
        dnnPrimitive_t conversion, void *from, void *to) {
    return dnnConversionExecute_F64(conversion, from, to);
}

dnnError_t dnnSumCreate(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, real *coefficients) {
    return dnnSumCreate_F64(pSum, attributes, nSummands, layout, coefficients);
}

dnnError_t dnnConcatCreate(
         dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src) {
    return dnnConcatCreate_F64(pConcat, attributes, nSrcTensors, src);
}

dnnError_t dnnSplitCreate(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]) {
    return dnnSplitCreate_F64(pSplit, attributes, nDstTensors, layout, dstChannelSize);
}

dnnError_t dnnScaleCreate(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real alpha) {
    return dnnScaleCreate_F64(pScale, attributes, dataLayout, alpha);
}

dnnError_t dnnConvolutionCreateForward(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateForward_F64(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateForwardBias_F64(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateBackwardData_F64(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateBackwardFilter_F64(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]) {
    return dnnConvolutionCreateBackwardBias_F64(pConvolution, attributes, algorithm,
        dimension, dstSize);
}

dnnError_t dnnGroupsConvolutionCreateForward(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateForward_F64(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateForwardBias_F64(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateBackwardData_F64(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateBackwardFilter_F64(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]) {
    return dnnGroupsConvolutionCreateBackwardBias_F64(pConvolution, attributes, algorithm,
        groups, dimension, dstSize);
}

dnnError_t dnnReLUCreateForward(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real negativeSlope) {
    return dnnReLUCreateForward_F64(pRelu, attributes, dataLayout, negativeSlope);
}

dnnError_t dnnReLUCreateBackward(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, real negativeSlope) {
    return dnnReLUCreateBackward_F64(pRelu, attributes, diffLayout, dataLayout, negativeSlope);
}

dnnError_t dnnLRNCreateForward(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, real alpha, real beta, real k) {
    return dnnLRNCreateForward_F64(pLrn, attributes, dataLayout, kernel_size, alpha, beta, k);
}

dnnError_t dnnLRNCreateBackward(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, real alpha, real beta, real k) {
    return dnnLRNCreateBackward_F64(pLrn, attributes, diffLayout, dataLayout, kernel_size, alpha, beta, k);
}

dnnError_t dnnBatchNormalizationCreateForward(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real eps) {
    return dnnBatchNormalizationCreateForward_F64(pBatchNormalization, attributes, dataLayout, eps);
}

dnnError_t dnnBatchNormalizationCreateBackwardScaleShift(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real eps) {
    return dnnBatchNormalizationCreateBackwardScaleShift_F64(pBatchNormalization, attributes, dataLayout, eps);
}

dnnError_t dnnBatchNormalizationCreateBackwardData(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real eps) {
    return dnnBatchNormalizationCreateBackwardData_F64(pBatchNormalization, attributes, dataLayout, eps);
}

dnnError_t dnnPoolingCreateForward(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType) {
    return dnnPoolingCreateForward_F64(pPooling, attributes, op, srcLayout,
        kernelSize, kernelStride, inputOffset, borderType);
}

dnnError_t dnnPoolingCreateBackward(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType) {
    return dnnPoolingCreateBackward_F64(pPooling, attributes, op, srcLayout,
        kernelSize, kernelStride,inputOffset, borderType);
}

dnnError_t dnnInnerProductCreateForward(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateForward_F64(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateForwardBias(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateForwardBias_F64(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateBackwardData(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateBackwardData_F64(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateBackwardFilter(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateBackwardFilter_F64(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateBackwardBias(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]) {
    return dnnInnerProductCreateBackwardBias_F64(pInnerProduct, attributes, dimensions, dstSize);
}
#else
/*******************************************************************************
 * F32 section: single precision
 ******************************************************************************/
dnnError_t dnnLayoutCreate(
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]) {
    return dnnLayoutCreate_F32(pLayout, dimension, size, strides);
}

dnnError_t dnnLayoutCreateFromPrimitive(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type) {
    return dnnLayoutCreateFromPrimitive_F32(pLayout, primitive, type);
}

size_t dnnLayoutGetMemorySize(
        const dnnLayout_t layout) {
    return dnnLayoutGetMemorySize_F32(layout);
}

int dnnLayoutCompare(
        const dnnLayout_t l1, const dnnLayout_t l2) {
    return dnnLayoutCompare_F32(l1, l2);
}

dnnError_t dnnAllocateBuffer(
        void **pPtr, dnnLayout_t layout) {
    return dnnAllocateBuffer_F32(pPtr, layout);
}

dnnError_t dnnReleaseBuffer(
        void *ptr) {
    return dnnReleaseBuffer_F32(ptr);
}

dnnError_t dnnLayoutDelete(
        dnnLayout_t layout) {
    return dnnLayoutDelete_F32(layout);
}

dnnError_t dnnPrimitiveAttributesCreate(
        dnnPrimitiveAttributes_t *attributes) {
    return dnnPrimitiveAttributesCreate_F32(attributes);
}

dnnError_t dnnPrimitiveAttributesDestroy(
        dnnPrimitiveAttributes_t attributes) {
    return dnnPrimitiveAttributesDestroy_F32(attributes);
}

dnnError_t dnnPrimitiveGetAttributes(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes) {
    return dnnPrimitiveGetAttributes_F32(primitive, attributes);
}

dnnError_t dnnExecute(
        dnnPrimitive_t primitive, void *resources[]) {
    return dnnExecute_F32(primitive, resources);
}

dnnError_t dnnExecuteAsync(
        dnnPrimitive_t primitive, void *resources[]) {
    return dnnExecuteAsync_F32(primitive, resources);
}

dnnError_t dnnWaitFor(
        dnnPrimitive_t primitive) {
    return dnnWaitFor_F32(primitive);
}

dnnError_t dnnDelete(
        dnnPrimitive_t primitive) {
    return dnnDelete_F32(primitive);
}

dnnError_t dnnConversionCreate(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to) {
    return dnnConversionCreate_F32(pConversion, from, to);
}

dnnError_t dnnConversionExecute(
        dnnPrimitive_t conversion, void *from, void *to) {
    return dnnConversionExecute_F32(conversion, from, to);
}

dnnError_t dnnSumCreate(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, real *coefficients) {
    return dnnSumCreate_F32(pSum, attributes, nSummands, layout, coefficients);
}

dnnError_t dnnConcatCreate(
        dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src) {
    return dnnConcatCreate_F32(pConcat, attributes, nSrcTensors, src);
}

dnnError_t dnnSplitCreate(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]) {
    return dnnSplitCreate_F32(pSplit, attributes, nDstTensors, layout, dstChannelSize);
}

dnnError_t dnnScaleCreate(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real alpha) {
    return dnnScaleCreate_F32(pScale, attributes, dataLayout, alpha);
}

dnnError_t dnnConvolutionCreateForward(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateForward_F32(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateForwardBias_F32(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateBackwardData_F32(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnConvolutionCreateBackwardFilter_F32(pConvolution, attributes, algorithm,
        dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]) {
    return dnnConvolutionCreateBackwardBias_F32(pConvolution, attributes, algorithm, dimension, dstSize);
}

dnnError_t dnnGroupsConvolutionCreateForward(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateForward_F32(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateForwardBias_F32(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateBackwardData_F32(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType) {
    return dnnGroupsConvolutionCreateBackwardFilter_F32(pConvolution, attributes, algorithm,
        groups, dimension, srcSize, dstSize, filterSize, convolutionStrides, inputOffset, borderType);
}

dnnError_t dnnGroupsConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]) {
    return dnnGroupsConvolutionCreateBackwardBias_F32(pConvolution, attributes, algorithm,
        groups, dimension, dstSize);
}

dnnError_t dnnReLUCreateForward(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real negativeSlope) {
    return dnnReLUCreateForward_F32(pRelu, attributes, dataLayout, negativeSlope);
}

dnnError_t dnnReLUCreateBackward(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, real negativeSlope) {
    return dnnReLUCreateBackward_F32(pRelu, attributes, diffLayout, dataLayout, negativeSlope);
}

dnnError_t dnnLRNCreateForward(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, real alpha, real beta, real k) {
    return dnnLRNCreateForward_F32(pLrn, attributes, dataLayout, kernel_size, alpha, beta, k);
}

dnnError_t dnnLRNCreateBackward(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, real alpha, real beta, real k) {
    return dnnLRNCreateBackward_F32(pLrn, attributes, diffLayout, dataLayout, kernel_size, alpha, beta, k);
}

dnnError_t dnnBatchNormalizationCreateForward(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real eps) {
    return dnnBatchNormalizationCreateForward_F32(pBatchNormalization, attributes, dataLayout, eps);
}

dnnError_t dnnBatchNormalizationCreateBackwardScaleShift(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real eps) {
    return dnnBatchNormalizationCreateBackwardScaleShift_F32(pBatchNormalization, attributes, dataLayout, eps);
}

dnnError_t dnnBatchNormalizationCreateBackwardData(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, real eps) {
    return dnnBatchNormalizationCreateBackwardData_F32(pBatchNormalization, attributes, dataLayout, eps);
}

dnnError_t dnnPoolingCreateForward(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType) {
    return dnnPoolingCreateForward_F32(pPooling, attributes, op, srcLayout,
        kernelSize, kernelStride, inputOffset, borderType);
}

dnnError_t dnnPoolingCreateBackward(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType) {
    return dnnPoolingCreateBackward_F32(pPooling, attributes, op, srcLayout,
        kernelSize, kernelStride, inputOffset, borderType);
}

dnnError_t dnnInnerProductCreateForward(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateForward_F32(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateForwardBias(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateForwardBias_F32(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateBackwardData(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateBackwardData_F32(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateBackwardFilter(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels) {
    return dnnInnerProductCreateBackwardFilter_F32(pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

dnnError_t dnnInnerProductCreateBackwardBias(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]) {
    return dnnInnerProductCreateBackwardBias_F32(pInnerProduct, attributes, dimensions, dstSize);
}

#endif
  
}  // namespace paddle
