import tensorflow as tf

from flask import Flask, request, render_template
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

app = Flask(__name__)


@app.route('/')
def page():
    return render_template('index.html')


sess = tf.InteractiveSession()

learning_rate = 0.001
training_epochs = 15
batch_size = 100

TB_SUMMARY_DIR = './tb/mnist'

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(X, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)

keep_prob = tf.placeholder(tf.float32)


def setup_nn():
    with tf.variable_scope('layer1') as scope:
        W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([512]))
        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
        
        tf.summary.histogram("X", X)
        tf.summary.histogram("weights", W1)
        tf.summary.histogram("bias", b1)
        tf.summary.histogram("layer", L1)

    with tf.variable_scope('layer2') as scope:
        W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([512]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

        tf.summary.histogram("weights", W2)
        tf.summary.histogram("bias", b2)
        tf.summary.histogram("layer", L2)

    with tf.variable_scope('layer3') as scope:
        W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.random_normal([512]))
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        tf.summary.histogram("weights", W3)
        tf.summary.histogram("bias", b3)
        tf.summary.histogram("layer", L3)

    with tf.variable_scope('layer4') as scope:
        W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([512]))
        L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

        tf.summary.histogram("weights", W4)
        tf.summary.histogram("bias", b4)
        tf.summary.histogram("layer", L4)

    with tf.variable_scope('layer5') as scope:
        W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([10]))
        hypothesis = tf.matmul(L4, W5) + b5

        tf.summary.histogram("weights", W5)
        tf.summary.histogram("bias", b5)
        tf.summary.histogram("layer", hypothesis)

    res = tf.argmax(hypothesis, 1)

    # Define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    tf.summary.scalar("loss", cost)
    
    # Summary
    summary = tf.summary.merge_all()

    # Initialize
    sess.run(tf.global_variables_initializer())

    # Create summary writer
    writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)
    global_step = 0

    print('::: Start Learning! :::')

    # Train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('::: Learning Finished! :::')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    return X, res


X, result = setup_nn()


@app.route('/determine', methods=['POST'])
def determine():
    amts = [int(x, 16) / 255 for x in request.form['amounts'].split(',')]
    res = sess.run(result, feed_dict={X: [amts], keep_prob: 1.0})
    return str(res[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0')

