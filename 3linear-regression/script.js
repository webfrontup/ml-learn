import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

window.onload = async () => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];

    tfvis.render.scatterplot(
        { name: '线性回归训练集' },
        { values: xs.map((x, i) => ({ x, y: ys[i] })) },
        { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
    );

    const model = tf.sequential(); // 连续的模型
    // 全链接层  
    // units：神经元个数    inputShape：维数、特征数量
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    // 随时函数 均方误差
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });

    const inputs = tf.tensor(xs);
    const labels = tf.tensor(ys);
    await model.fit(inputs, labels, {
        batchSize: 4, // 每个梯度更新的样本数量
        epochs: 200, // 对训练数据数组进行迭代的次数 超参数
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练过程' },
            ['loss']
        )
    });
    // 预测
    const output = model.predict(tf.tensor([5]));
    console.log(`如果 x 为 5，那么预测 y 为 ${output.dataSync()[0]}`);
};