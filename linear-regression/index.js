import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {
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

    const inputs = tf.tensor(xs)
    const labels = tf.tensor(ys)
    await model.fit(inputs, labels, {
        batchSize: 4,//每次扔4个点数据
        epochs: 30,//训练30次
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    })

    //训练完后执行预测5的值
    const output = model.predict(tf.tensor([5]))//预测数据要是tensor数据
    console.log(output.print());//预测结果也是tensor数据
    //将tensor数据转换成一般js数据
    console.log(output.dataSync());//输入是什么数据类型，转换后的数据类型也是一样的，所以是数组。
    alert('预测x是5的时候，y的结果是：'+output.dataSync()[0]);
};
