import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {getData} from './data'

window.onload = async () => {
    const data = getData(400);
    console.log(data)

    tfvis.render.scatterplot(
        {name: '逻辑回归训练数据'},
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0)
            ]
        }
    );

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation: 'sigmoid' //激活函数sigmoid
    }))

};