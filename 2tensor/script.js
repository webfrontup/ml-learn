import * as tf from '@tensorflow/tfjs';

// 传统 for 循环
const input = [1, 2, 3, 4];
const w = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]];
const output = [0, 0, 0, 0];

for (let i = 0; i < w.length; i++) {
    for (let j = 0; j < input.length; j++) {
        output[i] += input[j] * w[i][j];
    }
}

console.log(output);

tf.tensor(w).dot(tf.tensor(input)).print();



tf.tensor(1) // shape: []
tf.tensor([1,2]) // shape: [2] 第一层有2个值
tf.tensor([[1],[3]]) // shape: [2,1] 第一层有两个值，第二层有一个值
tf.tensor([[1,2], [3,4]]); // shape: [2,2]
tf.tensor([[[1]]]); // shape: [1,1,1]


const a = tf.tensor1d([1, 2]);
const b = tf.tensor2d([[1, 2], [3, 4]]);
const c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);

b.print() // Tensor[[1,2],[3,4]];
a.dot(b).print(); // Tensor[(7, 10)];  // or tf.dot(a, b) => // [1*1+2*3, 1*2+2*4]
b.dot(a).print(); // Tensor[(5, 11)];  // [1*1+2*2,3*1+4*2]
b.dot(c).print(); // Tensor[([9, 12, 15], [19, 26, 33])]; 

const a = tf.tensor1d([1, 2]);
const b = tf.tensor2d([[3, 4], [5, 6]]);

a.dot(b).print(); // [1*3+2*5, 1*4+2*6] => [13, 16]


//const a = tf.tensor1d([3,4]);
//const b = tf.tensor2d([[5,6], [7,8]]);

//a.dot(b).print(); // [3*5+4*7, 3*6+4*8] => [43, 50]




// tensor2d
const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
x.print(); // Tensor[[1, 2, 3, 4], [5, 6, 7, 8]] 

x.slice([1,0],[1,2]).print() 
// [1,0]: 第一维的第二项[5, 6, 7, 8] 第二维的第0项 5
// [1,2]: 尺寸 第一维1个数值，第二维2个数值 [5,6]
// [[5,6],]



