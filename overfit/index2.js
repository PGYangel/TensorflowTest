import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {getData} from './data2'

window.onload = async () => {
    const data = getData(200);

    tfvis.render.scatterplot(
        {name:'训练数据'},
        {
            values:[
                data.filter(p=>p.label===1),
                data.filter(p=>p.label===0)
            ]
        }
    );

    const model = tf.sequential()
    model.add(tf.layers.dense({
        units:1,
        activation:'sigmoid',
        inputShape:[2]
    }))
    model.compile({
        loss:tf.losses.logLoss,
        optimizer:tf.train.adam(0.1)
    })

    const inputs =tf.tensor(data.map(p =>[p.x,p.y]))
    const labels =tf.tensor(data.map(p =>[p.label]))

    await model.fit(inputs,labels,{
        validationSplit:0.2,
        epochs:200,
        callbacks:tfvis.show.fitCallbacks(
            {name:'训练过程'},
            ['loss','val_loss'],
            {callbacks:['onEpochEnd']}
        )
    })


};
