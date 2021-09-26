import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

// hs data --cors

window.onload = async () => {
    const data = new MnistData();
    await data.load();
    const examples = data.nextTestBatch(20);
    const surface = tfvis.visor().surface({ name: '输入示例' });
    // 20张图片
    for (let i = 0; i < 20; i += 1) {
        const imageTensor = tf.tidy(() => { // 清除webGL中 没用的tensor内存
            return examples.xs
                .slice([i, 0], [1, 784]) // 第i个图片的第1个像素值，只切割一个图片，切割刀784个像素
                .reshape([28, 28, 1]); // 重新更改图片形状 宽28 高28 黑白图片1
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px';
        await tf.browser.toPixels(imageTensor, canvas);
        //document.body.appendChild(canvas)
        surface.drawArea.appendChild(canvas);
    }

    const model = tf.sequential();
    // 卷积层
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5, // 卷积核大小
        filters: 8, // 超参数
        strides: 1, // 移动步长
        activation: 'relu', // 如果小于0则舍弃，反之使用
        kernelInitializer: 'varianceScaling' // 卷积核初始化
    }));
    // 最大池化 层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2], // 池化尺寸 2x2
        strides: [2, 2]
    }));
    // 卷积层 再次提取特征
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.flatten()); // 把高维数据 摊平 放到一维数组里
    // 最后全链接层 分类
    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    }));
    model.compile({
		loss: "categoricalCrossentropy", // 交叉熵损失函数处理多分类问题
		optimizer: tf.train.adam(),
		metrics: ["accuracy"], // 准确度度量
	});

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(1000);
        return [
            d.xs.reshape([1000, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(200);
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ];
    });

    //进行训练
    await model.fit(trainXs, trainYs, {
        validationData: [testXs, testYs],
        batchSize: 500,
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    const canvas = document.querySelector('canvas');

    canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.fillRect(e.offsetX, e.offsetY, 25, 25);
        }
    });

    window.clear = () => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0, 0, 300, 300);
    };

    clear();

    window.predict = () => {
        const input = tf.tidy(() => {
            return tf.image.resizeBilinear(
                tf.browser.fromPixels(canvas),
                [28, 28],
                true
            ).slice([0, 0, 0], [28, 28, 1])
            .toFloat()
            .div(255)
            .reshape([1, 28, 28, 1]);
        });
        const pred = model.predict(input).argMax(1);
        console.log(`预测结果为 ${pred.dataSync()[0]}`);
    };
};


// 池化层 用于提取最强的特征
// 扩大感受视野，减少计算量
// 池化层没有权重需要训练

// 全链接层
// 作为输出层 作为分类器 全链接层权重需要训练



