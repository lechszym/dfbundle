import tensorflow as tf
import numpy as np
import sys

tf.compat.v1.disable_eager_execution()

def df_bundle(model, x, batch_size=1000):
    """
    df_bundle computes the tangent bundle of a model based on input x

    :param model: tensorflow model of M inputs and K outputs
    :param x: set of N inputs for which the tangent bundle will be computed
    :return: NxM+1xK set of tangents corresponding to N inputs and K outputs
    """
    N = len(x)
    K = model.layers[-1].output.shape[1]

    v = model.output.op.inputs[0]
    v_split = tf.split(v, num_or_size_splits=K, axis=1)

    v_out = np.zeros((N,K))
    iterate = tf.keras.backend.function([model.input], v)

    i=0
    while(i < N):
        j = i+batch_size
        if j>N:
            j=N

        v_out[i:j] = iterate(x[i:j])
        i = j

    ws = list()
    for k in range(K):
        grads = tf.keras.backend.gradients(v_split[k], model.input)[0]
        iterate = tf.keras.backend.function([model.input], grads)

        ws.append(np.zeros(x.shape))
        i = 0
        while (i < N):
            j = i + batch_size
            if j > N:
                j = N

            ws[k][i:j] = iterate(x[i:j])
            i = j

        w0 = np.expand_dims(v_out[:,k]-np.sum(np.reshape(x*ws[k],(N,-1)),axis=1),axis=1)
        ws[k] = np.expand_dims(np.concatenate((w0,np.reshape(ws[k],(N,-1))),axis=1),axis=2)

    ws = np.concatenate(ws,axis=2)

    return ws

def df_similarity(x,ws,tensordot_datasize_limit=10000):
    """
    df_similarity computes similarity matrix for a tangent bundle

    :param x: NxM set of N inputs of dimension M to evaluate similarity on
    :param ws: NxM+1xK tangent bundle to evaluate similarity on
    :return: NxN similarity matrix based on
    """
    N = len(x)
    K = ws.shape[2]

    if N<=tensordot_datasize_limit:
        V = np.tensordot(np.reshape(x,(N,-1)), ws[:, 1:, :], axes=[[1], [1]]) + np.expand_dims(ws[:, 0, :], axis=0)
        S = np.ones((N, N))

        if K == 1:
            V = V[:, :, 0]
            V[V < 0] = 0
            V[V > 0] = 1
        else:
            V = np.argmax(V, axis=2)

        for n in range(N):
            S[n, :] = np.mean(np.expand_dims(V[:, n], axis=1) == V, axis=0)

    else:
        batch_size = int(tensordot_datasize_limit/K)
        V = np.zeros((N,N),dtype='uint8')
        S = np.zeros((N, N))
        nBatches = int(np.ceil(len(x)/batch_size))
        for i in range(nBatches):
            b1s = i*batch_size
            b1e = b1s+batch_size
            if b1e > N:
                b1e = N

            V1 = np.tensordot(np.reshape(x, (N, -1)), ws[b1s:b1e, 1:, :], axes=[[1], [1]]) + np.expand_dims(ws[b1s:b1e, 0, :],axis=0)

            if K == 1:
                V1 = V1[:, :, 0]
                V1[V1 < 0] = 0
                V1[V1 > 0] = 1
            else:
                V[:,b1s:b1e] = np.argmax(V1, axis=2)

        batch_size = int(tensordot_datasize_limit*tensordot_datasize_limit/N/N*100)
        if batch_size < 1:
            batch_size = 1
            nBatches = N
        else:
            nBatches = int(np.floor(N/batch_size))

        for i in range(nBatches):
            b1s = i*batch_size
            b1e = b1s+batch_size
            if b1e > N:
                b1e = N

            S += np.sum((np.expand_dims(V[b1s:b1e],axis=1)==np.expand_dims(V[b1s:b1e],axis=2)),axis=0)

        S /= N
    return S

def df_entropy(model,x,verbose=False):
    """
    df_entropy computes the conceptual capacity of a model based on data sample

    :param model: tensorflow model of M inputs and K outputs
    :param x: set of N inputs for evaluation of conceptual capacity
    :return: conceptual capacity of the model evaluated on x
    """

    if verbose:
        sys.stdout.write("Computing the df bundle for %d points..." % len(x))
        sys.stdout.flush()

    ws = df_bundle(model,x)

    if verbose:
        sys.stdout.write("done\n")
        sys.stdout.write("Creating the %dx%d similarity matrix..." % (len(x),len(x)))
        sys.stdout.flush()

    S = df_similarity(x,ws)

    
    if verbose:
        sys.stdout.write("done\n")
        sys.stdout.write("Evaluating the entropy of the similarity matrix...")
        sys.stdout.flush()

    lbd = np.linalg.eigvalsh(S)
    lbd /= np.sum(lbd)
    lbd = lbd[lbd > 1e-8]
    if len(lbd) == 1:
        H = 0
    else:
        H = -np.sum(lbd * np.log2(lbd))

    if verbose:
        sys.stdout.write("done\n")
        sys.stdout.flush()

    return H

