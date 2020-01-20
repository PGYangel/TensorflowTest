import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = () => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];

    tfvis.render.scatterplot(
        {name: '线性回归样本'},
        {
            values: xs.map((x, i) => ({x, y: ys[i]}))
        },
        {xAxisDomain: [0, 5], yAxisDomain: [0, 8]}
    );

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }));
    model.compile({
        loss: tf.losses.meanSquaredError,   //损失函数
        optimizer: tf.train.sgd(0.1)        //优化器：随机梯度下降法，0.1是学习速率，真实运用是试出来最优速率
    })

};
