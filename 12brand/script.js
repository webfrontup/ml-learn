import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getInputs } from './data';
import { img2x, file2img } from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';
const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];

window.onload = async () => {
	const { inputs, labels } = await getInputs();
	const surface = tfvis
		.visor()
		.surface({ name: "输入示例", styles: { height: 250 } });
	inputs.forEach((img) => {
		surface.drawArea.appendChild(img);
	});

	const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
	mobilenet.summary();
	//基于此阶段模型的输出;
	const layer = mobilenet.getLayer("conv_pw_13_relu");
	//训练一个新的模型
	const truncatedMobilenet = tf.model({
		inputs: mobilenet.inputs,
		outputs: layer.output,
	});
	//console.log(layer.outputShape); [null, 7, 7, 256]
	const model = tf.sequential(); // 连续的模型
	model.add(
		//一维向量
		tf.layers.flatten({
			inputShape: layer.outputShape.slice(1), //[7, 7, 256]
		})
	);
	// 添加神经网络
	model.add(
		tf.layers.dense({
			units: 10,
			activation: "relu",
		})
	);
	// 分类
	// 添加神经网络
	model.add(
		tf.layers.dense({
			units: NUM_CLASSES,
			activation: "softmax",
		})
	);
    // 随时函数优化器
	model.compile({
		loss: "categoricalCrossentropy", // 交叉熵
		optimizer: tf.train.adam(),
	});

	// 训练数据经过截断模型，转为可以用于新模型训练的数据
	const { xs, ys } = tf.tidy(() => {
		const xs = tf.concat(
			inputs.map((imgEl) => truncatedMobilenet.predict(img2x(imgEl)))
		);
		const ys = tf.tensor(labels);
		return { xs, ys };
	});

	await model.fit(xs, ys, {
		epochs: 20,
		callbacks: tfvis.show.fitCallbacks({ name: "训练效果" }, ["loss"], {
			callbacks: ["onEpochEnd"],
		}),
	});

	window.predict = async (file) => {
		const img = await file2img(file);
		document.body.appendChild(img);
		const pred = tf.tidy(() => {
			const x = img2x(img);
			const input = truncatedMobilenet.predict(x);
			return model.predict(input);
		});

		const index = pred.argMax(1).dataSync()[0];
		setTimeout(() => {
			console.log(`预测结果：${BRAND_CLASSES[index]}`);
		}, 0);
	};

	window.download = async () => {
		await model.save("downloads://model");
	};
};