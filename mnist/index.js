import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {MnistData} from './data'

window.onload =async ()=>{
    const data =new MnistData();
    await data.load();
    const examples =data.nextTestBatch(20);
    console.log(examples)
}
