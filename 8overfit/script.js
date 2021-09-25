import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data';

window.onload = async () => {
    // variance 噪音 干扰项
    const data = getData(200, 2);

    tfvis.render.scatterplot(
        { name: '训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [2],
        activation: "tanh",
        // kernelRegularizer: tf.regularizers.l2({ l2: 1 }) // 权重衰减
    }));
    model.add(tf.layers.dropout({ rate: 0.9 })); // 丢弃层，丢弃法
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        validationSplit: 0.2, // 分出20%作为验证集
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss'],
            { callbacks: ['onEpochEnd'] }
        )
    });
};

// 过拟合 解决方法
// 早停法 当发现 验证集的损失不降反增，则需要停下训练 直接停下
// 权重衰减 
// 丢弃法 