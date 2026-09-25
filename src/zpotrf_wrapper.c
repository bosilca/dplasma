/*
 * Copyright (c) 2010-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/types.h"
#include "dplasma/types_lapack.h"
#include "dplasmaaux.h"
#include "potrf_gpu_workspaces.h"

#include "zpotrf_U.h"
#include "zpotrf_L.h"
#include "cores/dplasma_plasmatypes.h"

#define MAX_SHAPES 1

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zpotrf_setrecursive - Set the recursive size parameter to enable
 *  recursive DAGs.
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to modify.
 *          On exit, the modified taskpool.
 *
 * @param[in] hmb
 *          The tile size to use for the smaller recursive call.
 *          hmb must be > 0, otherwise nothing is changed.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf
 *
 ******************************************************************************/
void
dplasma_zpotrf_setrecursive( parsec_taskpool_t *tp, int hmb )
{
    parsec_zpotrf_L_taskpool_t *parsec_zpotrf = (parsec_zpotrf_L_taskpool_t*)tp;
    if (hmb > 0) {
        parsec_zpotrf->_g_smallnb = hmb;
    }
}

#if defined(DPLASMA_HAVE_CUDA)
#include <cusolverDn.h>

static void *zpotrf_create_cuda_workspace(void *obj, void *user)
{
    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t *)obj;
    cusolverDnHandle_t cusolverDnHandle;
    cusolverStatus_t status;
    parsec_zpotrf_U_taskpool_t *tp = (parsec_zpotrf_U_taskpool_t*)user;
    dplasma_potrf_gpu_workspaces_t *wp = NULL;
    void *tmpmem, *host_buffer;
    size_t host_size;
    int workspace_size;
    int mb = tp->_g_descA->mb;
    int nb = tp->_g_descA->nb;
    size_t elt_size = sizeof(cuDoubleComplex);
    dplasma_enum_t uplo = tp->_g_uplo;

    status = cusolverDnCreate(&cusolverDnHandle);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    (void)status;

    status = cusolverDnZpotrf_bufferSize(cusolverDnHandle, dplasma_cublas_fill(uplo), nb, NULL, mb, &workspace_size);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cusolverDnDestroy(cusolverDnHandle);

    /* One scratch per device is on purpose, even though the cuSOLVER handle that
     * consumes it is per stream: potrf_zpotrf(k) reads T from potrf_zherk(k-1, k), so a
     * taskpool never has two panel factorizations in flight, and each taskpool registers
     * its own info key. Concurrent streams therefore cannot collide here. Should that
     * chain ever be relaxed, this has to become a per-stream info instead. */
    /* The scratch is allocated outside the zone PaRSEC manages for tiles. The zone can
     * legitimately be saturated by data copies, and it is only drained by tasks
     * completing, so a task that needs scratch to run must not depend on it. */
    if( PARSEC_SUCCESS != gpu_device->memory_allocate(gpu_device,
                                                      workspace_size * elt_size,
                                                      &tmpmem) )
        return NULL;

    /* cuSOLVER needs its status argument to be device accessible, but not device
     * resident: a mapped allocation lets the GPU write the panel status straight into
     * host memory, so there is no copy to enqueue on the stream and no host pointer to
     * synchronize on. Copying into the user's INFO instead would be a device to host
     * transfer into pageable memory, which blocks the calling thread until the stream
     * drains. One slot per panel keeps a later successful panel from erasing an earlier
     * failure; zpotrf_destroy_cuda_workspace folds them into INFO. */
    host_size = (size_t)(dplasmaUpper == uplo ? tp->_g_descA->nt : tp->_g_descA->mt) * sizeof(int);
    if( cudaSuccess != cudaHostAlloc(&host_buffer, host_size, cudaHostAllocMapped) ) {
        gpu_device->memory_free(gpu_device, tmpmem);
        return NULL;
    }
    memset(host_buffer, 0, host_size);

    wp = (dplasma_potrf_gpu_workspaces_t*)malloc(sizeof(dplasma_potrf_gpu_workspaces_t));
    wp->tmpmem = tmpmem;
    wp->lwork = workspace_size;
    wp->gpu_device = gpu_device;
    wp->params = tp;
    wp->host_size = host_size;
    wp->host_buffer = host_buffer;

    return wp;
}

static void zpotrf_destroy_cuda_workspace(void *_ws, void *_n)
{
    dplasma_potrf_gpu_workspaces_t *ws = (dplasma_potrf_gpu_workspaces_t*)_ws;
    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t*)ws->gpu_device;
    parsec_zpotrf_U_taskpool_t *tp = (parsec_zpotrf_U_taskpool_t*)ws->params;
    int *panel_info = (int*)ws->host_buffer;
    int nb_panels = (int)(ws->host_size / sizeof(int));
    int panel_stride = (dplasmaUpper == tp->_g_uplo) ? tp->_g_descA->nb : tp->_g_descA->mb;

    /* Every device holds the statuses of the panels it ran, and the destructors run in
     * an arbitrary order, so report the smallest failing k rather than letting the last
     * destructor win. Scanning upwards, the first non-zero entry is this device's
     * smallest. The value matches the CPU body, a row index into the whole matrix. */
    for( int k = 0; k < nb_panels; k++ ) {
        if( 0 == panel_info[k] ) continue;
        int candidate = k * panel_stride + panel_info[k];
        if( 0 == *tp->_g_INFO || candidate < *tp->_g_INFO )
            *tp->_g_INFO = candidate;
        break;
    }

    cudaFreeHost(ws->host_buffer);
    gpu_device->memory_free(gpu_device, ws->tmpmem);
    free(ws);
    (void)_n;
}
#endif

#if defined(DPLASMA_HAVE_HIP)
static void *zpotrf_create_hip_workspace(void *obj, void *user)
{
    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t *)obj;
    dplasma_potrf_gpu_workspaces_t *wp = NULL;
    void *tmpmem;
    (void)user;

    /* See zpotrf_create_cuda_workspace for why this bypasses the tile zone and why a
     * single scratch per device is enough. */
    if( PARSEC_SUCCESS != gpu_device->memory_allocate(gpu_device, sizeof(int), &tmpmem) )
        return NULL;

    wp = (dplasma_potrf_gpu_workspaces_t*)malloc(sizeof(dplasma_potrf_gpu_workspaces_t));
    wp->tmpmem = tmpmem;
    wp->lwork = 0;
    wp->gpu_device = gpu_device;
    /* rocSOLVER still writes its status into device memory and nobody reads it back;
     * see zpotrf_destroy_cuda_workspace for what the CUDA path does instead. */
    wp->params = NULL;
    wp->host_size = 0;
    wp->host_buffer = NULL;

    return wp;
}

static void zpotrf_destroy_hip_workspace(void *_ws, void *_n)
{
    dplasma_potrf_gpu_workspaces_t *ws = (dplasma_potrf_gpu_workspaces_t*)_ws;
    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t*)ws->gpu_device;
    gpu_device->memory_free(gpu_device, ws->tmpmem);
    free(ws);
    (void)_n;
}
#endif

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf_New - Generates the taskpool that Computes the Cholesky
 * factorization of a symmetric positive definite (or Hermitian positive
 * definite in the complex case) matrix A, with or without recursive calls.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = dplasmaLower}^{U^H\times U, if uplo = dplasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 * WARNING: The computations are not done by this call.
 *
 * If you want to enable the recursive DAGs, don't forget to set the recursive
 * tile size and to synchonize the taskpool ids after the computations since those
 * are for now local. You can follow the code of dplasma_zpotrf_rec() as an
 * example to do this.
 *
 * Hierarchical DAG Scheduling for Hybrid Distributed Systems; Wu, Wei and
 * Bouteiller, Aurelien and Bosilca, George and Faverge, Mathieu and Dongarra,
 * Jack. 29th IEEE International Parallel & Distributed Processing Symposium,
 * May 2015. (https://hal.inria.fr/hal-0107835)
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A is referenced;
 *          = dplasmaLower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 * @param[out] info
 *          Address where to store the output information of the factorization,
 *          this is not synchronized between the nodes, and might not be set
 *          when function exists.
 *          On DAG completion:
 *              - info = 0 on all nodes if successful.
 *              - info > 0 if the leading minor of order i of A is not positive
 *                definite, so the factorization could not be completed, and the
 *                solution has not been computed. Info will be equal to i on the
 *                node that owns the diagonal element (i,i), and 0 on all other
 *                nodes.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zpotrf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf_New
 * @sa dplasma_dpotrf_New
 * @sa dplasma_spotrf_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zpotrf_New( dplasma_enum_t uplo,
                    parsec_tiled_matrix_t *A,
                    int *info )
{
    parsec_zpotrf_L_taskpool_t *parsec_zpotrf = NULL;
#if defined(DPLASMA_HAVE_CUDA) || defined(DPLASMA_HAVE_HIP)
    char workspace_info_name[64];
    static int uid = 0;
#endif
    parsec_taskpool_t *tp = NULL;
    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection(A);

    /* Check input arguments */
    if ((uplo != dplasmaUpper) && (uplo != dplasmaLower)) {
        dplasma_error("dplasma_zpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    *info = 0;

    if ( uplo == dplasmaUpper ) {
        tp = (parsec_taskpool_t*)parsec_zpotrf_U_new( uplo, ddc_A, info);
    } else {
        tp = (parsec_taskpool_t*)parsec_zpotrf_L_new( uplo, ddc_A, info);
    }
    parsec_zpotrf =  (parsec_zpotrf_L_taskpool_t*)tp;

    parsec_zpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == parsec_zpotrf->_g_PRI_CHANGE)
      parsec_zpotrf->_g_PRI_CHANGE = A->nt;
#if defined(DPLASMA_HAVE_CUDA)
    /* It doesn't cost anything to define these infos if we have CUDA but
     * don't have GPUs on the current machine, so we do it non-conditionally */
    parsec_zpotrf->_g_cuda_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::CUDA::HANDLES", NULL);
    snprintf(workspace_info_name, 64, "DPLASMA::ZPOTRF(%d)::WS", uid++);
    parsec_zpotrf->_g_cuda_workspaces_infokey = parsec_info_register(&parsec_per_device_infos, workspace_info_name,
                                                           zpotrf_destroy_cuda_workspace, NULL,
                                                           zpotrf_create_cuda_workspace, parsec_zpotrf,
                                                           NULL);
#else
    parsec_zpotrf->_g_cuda_handles_infokey = PARSEC_INFO_ID_UNDEFINED;
    parsec_zpotrf->_g_cuda_workspaces_infokey = PARSEC_INFO_ID_UNDEFINED;
#endif
#if defined(DPLASMA_HAVE_HIP)
    /* It doesn't cost anything to define these infos if we have HIP but
     * don't have GPUs on the current machine, so we do it non-conditionally */
    parsec_zpotrf->_g_hip_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::HIP::HANDLES", NULL);
    snprintf(workspace_info_name, 64, "DPLASMA::ZPOTRF(%d)::WS", uid++);
    parsec_zpotrf->_g_hip_workspaces_infokey = parsec_info_register(&parsec_per_device_infos, workspace_info_name,
                                                           zpotrf_destroy_hip_workspace, NULL,
                                                           zpotrf_create_hip_workspace, parsec_zpotrf,
                                                           NULL);
#else
    parsec_zpotrf->_g_hip_handles_infokey = PARSEC_INFO_ID_UNDEFINED;
    parsec_zpotrf->_g_hip_workspaces_infokey = PARSEC_INFO_ID_UNDEFINED;
#endif
    int shape = 0;
    dplasma_setup_adtt_all_loc( ddc_A,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    assert(shape == MAX_SHAPES);
    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zpotrf_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zpotrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf
 *
 ******************************************************************************/
void
dplasma_zpotrf_Destruct( parsec_taskpool_t *tp )
{
    parsec_zpotrf_L_taskpool_t *parsec_zpotrf = (parsec_zpotrf_L_taskpool_t *)tp;
    dplasma_clean_adtt_all_loc(parsec_zpotrf->_g_ddescA, MAX_SHAPES);
    dplasma_data_collection_t * ddc_A = parsec_zpotrf->_g_ddescA;

#if defined(DPLASMA_HAVE_CUDA)
    parsec_info_unregister(&parsec_per_device_infos, parsec_zpotrf->_g_cuda_workspaces_infokey, NULL);
#endif
#if defined(DPLASMA_HAVE_HIP)
    parsec_info_unregister(&parsec_per_device_infos, parsec_zpotrf->_g_hip_workspaces_infokey, NULL);
#endif

    parsec_taskpool_free(tp);
    /* free the dplasma_data_collection_t */
    dplasma_unwrap_data_collection(ddc_A);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf - Computes the Cholesky factorization of a symmetric positive
 * definite (or Hermitian positive definite in the complex case) matrix A.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = dplasmaLower}^{U^H\times U, if uplo = dplasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A is referenced;
 *          = dplasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf
 * @sa dplasma_dpotrf
 * @sa dplasma_spotrf
 *
 ******************************************************************************/
int
dplasma_zpotrf( parsec_context_t *parsec,
                dplasma_enum_t uplo,
                parsec_tiled_matrix_t *A )
{
    parsec_taskpool_t *parsec_zpotrf = NULL;
    int info = 0, ginfo = 0 ;

    parsec_zpotrf = dplasma_zpotrf_New( uplo, A, &info );

    if ( parsec_zpotrf != NULL )
    {
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_zpotrf);
        dplasma_wait_until_completion(parsec);
        dplasma_zpotrf_Destruct( parsec_zpotrf );
    }

    /* This covers both cases when we have not compiled with MPI, or we don't need to do the reduce */
    ginfo = info;
#if defined(PARSEC_HAVE_MPI)
    /* If we don't need to reduce, don't do it, this way we don't require MPI to be initialized */
    if( A->super.nodes > 1 )
        MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, *(MPI_Comm*)dplasma_pcomm);
#endif

    return ginfo;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf_rec - Computes the Cholesky factorization of a symmetric
 * positive definite (or Hermitian positive definite in the complex case) matrix
 * An, using the recursive DAGs feature if hmb is smaller than A.mb or A.nb.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = dplasmaLower}^{U^H\times U, if uplo = dplasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A is referenced;
 *          = dplasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 * @param[in] hmb
 *          The tile size to use for the smaller recursive call.
 *          If hmb <= 0 or hmb > A.mb, the classic algorithm without recursive
 *          calls is applied.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf
 * @sa dplasma_dpotrf
 * @sa dplasma_spotrf
 *
 ******************************************************************************/
int
dplasma_zpotrf_rec( parsec_context_t *parsec,
                    dplasma_enum_t uplo,
                    parsec_tiled_matrix_t *A, int hmb )
{
    parsec_taskpool_t *parsec_zpotrf = NULL;
    int info = 0, ginfo = 0 ;

    parsec_zpotrf = dplasma_zpotrf_New( uplo, A, &info );
    if ( parsec_zpotrf != NULL )
    {
        dplasma_zpotrf_setrecursive( (parsec_taskpool_t*)parsec_zpotrf, hmb );
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_zpotrf);
        dplasma_wait_until_completion(parsec);
        dplasma_zpotrf_Destruct( parsec_zpotrf );
        parsec_taskpool_sync_ids(); /* recursive DAGs are not synchronous on ids */
    }

    /* This covers both cases when we have not compiled with MPI, or we don't need to do the reduce */
    ginfo = info;
#if defined(PARSEC_HAVE_MPI)
    /* If we don't need to reduce, don't do it, this way we don't require MPI to be initialized */
    if( A->super.nodes > 1 )
        MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, *(MPI_Comm*)dplasma_pcomm);
#endif
    return ginfo;
}
