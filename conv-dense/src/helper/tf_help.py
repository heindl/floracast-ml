"""
Helpful functions for dealing with Tensorflow.
"""

def prep_feed_dict(model_tensors, X_batch, y_batch, pool_size, is_train=True):
    # compute padding for current batch (applied after conv layer)
        pads = [[0, 0], [0, (pool_size - len(X_batch[0]) % pool_size)%pool_size], [0, 0], [0, 0]]

        # compute size of maxpool height step
        if len(X_batch[0]) % pool_size == 0:
            step = len(X_batch[0])/pool_size
        else:
            step = int(len(X_batch[0])/pool_size) + 1

        feed_dict = {
                model_tensors['X_t']: X_batch,
                model_tensors['y_t']: y_batch,
                model_tensors['pool_k_size_t']: [1, step, 1, 1],
                model_tensors['pool_strides_t']: [1, step, 1, 1],
                model_tensors['pool_pad_t']: pads,
                model_tensors['apply_dropout']: is_train}

        return feed_dict


def compute_train_step(sess, model, model_tensors, Xb, optimizer, pool_size):
    acc_epoch = 0
    loss_epoch = 0
    num_examples = 0
    embed_epoch, y_epoch = [], []

    for  i in range(len(Xb[0])):
        # get all features of the current fixed length
        X_batch,y_batch = Xb[0][i],Xb[1][i]

        feed_dict = prep_feed_dict(model_tensors, X_batch, y_batch, pool_size, is_train=True)

        _, acc_batch, loss_batch, embed_batch = sess.run([optimizer, model.eval.accuracy, model.eval.loss, model.embedding], feed_dict=feed_dict)

        # append embeddings and labels to single list
        embed_epoch.extend(embed_batch) 
        y_epoch.extend(y_batch)  

        # compute running weighted accuracy and loss
        num_examples += len(X_batch)
        acc_epoch += acc_batch * len(X_batch)
        loss_epoch += loss_batch * len(X_batch)

    acc_epoch = acc_epoch / float(num_examples)
    loss_epoch = loss_epoch / float(num_examples)

    return acc_epoch, loss_epoch, embed_epoch, y_epoch


def compute_eval_step(sess, model, model_tensors, Xb, pool_size):
    acc_all = 0
    num_examples = 0
    embed_all, y_all = [], []

    # iterate over each batch
    for i in range(len(Xb[0])):
        X_batch,y_batch = Xb[0][i],Xb[1][i]

        # compute acc and embeddings
        feed_dict = prep_feed_dict(model_tensors, X_batch, y_batch, pool_size, is_train=False)
        acc_batch, embed_batch = sess.run([model.eval.accuracy, model.embedding], feed_dict=feed_dict)

        # append embeddings and labels to single list
        embed_all.extend(embed_batch) 
        y_all.extend(y_batch)  

        # compute running weighted accuracy
        num_examples += len(X_batch)
        acc_all += acc_batch * len(X_batch)

    acc_all = acc_all / float(num_examples)

    return acc_all, embed_all, y_all




