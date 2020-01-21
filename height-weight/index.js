import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {
    const heights = [150, 160, 170];
    const weights = [40, 50, 60];

    tfvis.render.scatterplot(
        {name: '身高体重训练数据'},
        {values: heights.map((x, i) => ({x, y: weights[i]}))},
        {xAxisDomain: [140, 180], yAxisDomain: [30, 70]}
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

    //归一化操作
    const inputs = tf.tensor(heights).sub(150).div(20);
    inputs.print();
    const labels = tf.tensor(weights).sub(40).div(20);
    labels.print();

    await model.fit(inputs, labels, {
        batchSize: 3,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    });

    //因为模型训练的时候导入的数据是归一化后的数据，所以预测数据也需要归一化
    const output = model.predict(tf.tensor([180]).sub(150).div(20))
    //输出的output是归一化数据，最终显示需要反归一化显示到前端
    alert('预测身高180cm的体重是：'+output.mul(20).add(40).dataSync()[0]+'kg')

};
