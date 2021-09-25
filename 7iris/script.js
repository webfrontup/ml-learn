import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';
// 花瓣花萼的长度和宽度+输出类别
// 输出类别：0山鸢尾 1变色鸢尾 2维吉尼亚鸢尾

window.onload = async () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

    // 初始化一个神经网络模型 为神经网络添加2个层
    // 设计层的神经元个数、inputShape、激活函数
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: 'sigmoid'
    }));
    // 保证输出的3个数的和为1， 则需要用 softmax 方法
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }));
    // 交叉熵损失函数处理多分类问题
    // 与准确度度量
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.1),
        metrics: ['accuracy']
    });

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest], // 验证集
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'], // 验证集的损失 验证集 准确度 验证集的准确度
            { callbacks: ['onEpochEnd'] } // 只展示 onEpochEnd
        )
    });

    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1,
        ]]);
        const pred = model.predict(input);
        console.log(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
    };
};