import './style.css'

//define brain.js
import * as brain from 'https://cdn.skypack.dev/brain.js'

const config = {
  binaryThresh: 0.5,
  hiddenLayers: [3,3,3,3], // array of ints for the sizes of the hidden layers in the network
  activation: 'sigmoid', // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
  leakyReluAlpha: 0.01, // supported for activation type 'leaky-relu'
}

// create a neural network with 2 inputs, 3 hidden layers, and 1 output
const net = new brain.NeuralNetwork({ config })

// train the network
net.train([
  { input: {r: 0, g:0, b:0}, output: { Black: 1} },
  { input: {r: 0.02, g:0.02, b:0.02}, output: { Black: 1} },
  { input: {r: 0.01, g:0.01, b:0.01}, output: { Black: 1} },
  { input: {r: 0.100, g:0.100, b:0.100}, output: { Gray: 1} },
  { input: {r: 0.060, g:0.060, b:0.060}, output: { Gray: 1} },
  { input: {r: 0.128, g:0.128, b:0.128}, output: { Gray: 1} },
  { input: {r: 0.140, g:0.140, b:0.140}, output: { Gray: 1} },
  { input: {r: 0.160, g:0.160, b:0.160}, output: { Gray: 1} },
  { input: {r: 0.071, g:0.063, b:0.064}, output: { Gray: 1} },
  { input: {r: 0.200, g:0.200, b:0.200}, output: { Gray: 1} },
  { input: {r: 0.195, g:0.204, b:0.202}, output: { Gray: 1} },
  { input: {r: 0.172, g:0.171, b:0.178}, output: { Gray: 1} },
  { input: {r: 0.214, g:0.211, b:0.208}, output: { Gray: 1} },
  { input: {r: 0.255, g:0, b:0}, output: { Red: 1} },
  { input: {r: 0.255, g:0, b:0.048}, output: { Red: 1} },
  { input: {r: 0.255, g:0.05, b:0.0}, output: { Red: 1} },
  { input: {r: 0.2, g:0.04, b:0.06}, output: { Red: 1} },
  { input: {r: 0.259, g:0.052, b:0.097}, output: { Red: 1} },
  { input: {r: 0.255, g:0.05, b:0.0}, output: { Red: 1} },
  { input: {r: 0.104, g:0.027, b:0.012}, output: { Red: 1} },
  { input: {r: 0.200, g:0.255, b:0.200}, output: { Green: 1} },
  { input: {r: 0.001, g:0.255, b:0.172}, output: { Green: 1} },
  { input: {r: 0.129, g:0.175, b:0.128}, output: { Green: 1} },
  { input: {r: 0.111, g:0.255, b:0.}, output: { Green: 1} },
  { input: {r: 0.217, g:0.255, b:0.221}, output: { Green: 1} },
  { input: {r: 0.149, g:0.255, b:0.007}, output: { Green: 1} },
  { input: {r: 0.177, g:0.255, b:0.041}, output: { Green: 1} },
  { input: {r: 0.0, g:0.255, b:0.0}, output: { Green: 1} },
  { input: {r: 0.0, g:0.255, b:0.216}, output: { Green: 1} },
  { input: {r: 0.118, g:0.255, b:0.211}, output: { Green: 1} },
  { input: {r: 0.0, g:0.211, b:0.205}, output: { Green: 1} },
  { input: {r: 0.071, g:0.169, b:0.181}, output: { Blue: 1} },
  { input: {r: 0.0, g:0.255, b:0.0}, output: { Green: 1} },
  { input: {r: 0.139, g:0.209, b:0.069}, output: { Green: 1} },
  { input: {r: 0.0, g:0.181, b:0.128}, output: { Green: 1} },
  { input: {r: 0.128, g:0.150, b:0.128}, output: { Green: 1} },
  { input: {r: 0.128, g:0.170, b:0.128}, output: { Green: 1} },
  { input: {r: 0.064, g:0.211, b:0.031}, output: { Green: 1} },
  { input: {r: 0.027, g:0.167, b:0.040}, output: { Green: 1} },
  { input: {r: 0.047, g:0.115, b:0.076}, output: { Green: 1} },
  { input: {r: 0.035, g:0.149, b:0.133}, output: { Green: 1} },
  { input: {r: 0.0, g:0.058, b:0.0}, output: { Green: 1} },
  { input: {r: 0.128, g:0.128, b:0.074}, output: { Green: 1} },
  { input: {r: 0.128, g:0.128, b:0.0}, output: { Green: 1} },
  { input: {r: 0.077, g:0.072, b:0.0}, output: { Green: 1} },
  { input: {r: 0.049, g:0.052, b:0.0}, output: { Green: 1} },
  { input: {r: 0.124, g:0.125, b:0.067}, output: { Green: 1} },
  { input: {r: 0.095, g:0.103, b:0.066}, output: { Green: 1} },
  { input: {r: 0.181, g:0.225, b:0.017}, output: { Green: 1} },
  { input: {r: 0.23, g:0.26, b:0.23}, output: { Green: 1} },
  { input: {r: 0.0, g:0.122, b:0.114}, output: { Green: 1} },
  { input: {r: 0.200, g:0.200, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.048, g:0.214, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.0, g:0.183, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.0, g:0.0, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.049, g:0.0, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.0, g:0.123, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.0, g:0.130, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.031, g:0.037, b:0.073}, output: { Blue: 1}},
  { input: {r: 0.024, g:0.033, b:0.066}, output: { Blue: 1}},
  { input: {r: 0.075, g:0.10, b:0.21}, output: { Blue: 1}},
  { input: {r: 0.092, g:0.147, b:0.207}, output: { Blue: 1}},
  { input: {r: 0.084, g:0.114, b:0.228}, output: { Blue: 1}},
  { input: {r: 0.200, g:0.100, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.128, g:0.128, b:0.232}, output: { Purple: 1}},
  { input: {r: 0.255, g:0.069, b:0.116}, output: { Purple: 1}},
  { input: {r: 0.234, g:0.058, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.128, g:0.000, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.128, g:0.031, b:0.075}, output: { Purple: 1}},
  { input: {r: 0.128, g:0.128, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.132, g:0.0, b:0.196}, output: { Purple: 1}},
  { input: {r: 0.128, g:0.0, b:0.128}, output: { Purple: 1}},
  { input: {r: 0.12, g:0.001, b:0.13}, output: { Purple: 1}},
  { input: {r: 0.062, g:0.035, b:0.080}, output: { Purple: 1}},
  { input: {r: 0.200, g:0.200, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.206, g:0.188, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.255, g:0.206, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.154, g:0.089, b:0.174}, output: { Purple: 1}},
  { input: {r: 0.205, g:0.001, b:0.255}, output: { Purple: 1}},
  { input: {r: 0.112, g:0.110, b:0.23}, output: { Purple: 1}},
  { input: {r: 0.158, g:0.24, b:0.211}, output: { Purple: 1}},
  { input: {r: 0.200, g:0.040, b:0.100}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.017, b:0.255}, output: { Pink: 1}},
  { input: {r: 0.183, g:0.128, b:0.128}, output: { Pink: 1}},
  { input: {r: 0.2, g:0.128, b:0.128}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.0, b:0.128}, output: { Pink: 1}},
  { input: {r: 0.205, g:0.128, b:0.128}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.201, b:0.222}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.0, b:0.255}, output: { Pink: 1}},
  { input: {r: 0.181, g:0.01, b:0.14}, output: { Pink: 1}},
  { input: {r: 0.238, g:0.072, b:0.214}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.104, b:0.224}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.128, b:0.128}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.149, b:0.137}, output: { Pink: 1}},
  { input: {r: 0.255, g:0.147, b:0.128}, output: { Pink: 1}},
  { input: {r: 0.24, g:0.23, b:0.23}, output: { Pink: 1}},
  { input: {r: 0.235, g:0.115, b:0.084}, output: { Red: 1}},
  { input: {r: 0.240, g:0.114, b:0.096}, output: { Red: 1}},
  { input: {r: 0.200, g:0.100, b:0.050}, output: { Brown: 1}},
  { input: {r: 0.124, g:0.072, b:0.0}, output: { Brown: 1}},
  { input: {r: 0.128, g:0.049, b:0.042}, output: { Brown: 1}},
  { input: {r: 0.191, g:0.064, b:0.059}, output: { Red: 1}},
  { input: {r: 0.082, g:0.007, b:0.041}, output: { Red: 1}},
  { input: {r: 0.200, g:0.100, b:0.050}, output: { Brown: 1}},
  { input: {r: 0.200, g:0.110, b:0.040}, output: { Brown: 1}},
  { input: {r: 0.109, g:0.047, b:0.016}, output: { Brown: 1}},
  { input: {r: 0.157, g:0.098, b:0.069}, output: { Brown: 1}},
  { input: {r: 0.064, g:0.001, b:0.059}, output: { Brown: 1}},
  { input: {r: 0.255, g:0.255, b:0.100}, output: { Yellow: 1}},
  { input: {r: 0.255, g:0.255, b:0.000}, output: { Yellow: 1}},
  { input: {r: 0.255, g:0.205, b:0.163}, output: { Yellow: 1}},
  { input: {r: 0.255, g:0.255, b:0.203}, output: { Yellow: 1}},
  { input: {r: 0.255, g:0.255, b:0.230}, output: { Yellow: 1}},
  { input: {r: 0.173, g:0.158, b:0.109}, output: { Yellow: 1}},
  { input: {r: 0.215, g:0.2, b:0.180}, output: { Yellow: 1}},
  { input: {r: 0.128, g:0.13, b:0.092}, output: { Yellow: 1}},
  { input: {r: 0.222, g:0.18, b:0.223}, output: { Yellow: 1}},
  { input: {r: 0.255, g:0.170, b:0.000}, output: { Orange: 1}},
  { input: {r: 0.24, g:0.11, b:0.000}, output: { Orange: 1}},
  { input: {r: 0.222, g:0.087, b:0.031}, output: { Orange: 1}},
  { input: {r: 0.255, g:0.255, b:0.255}, output: { White: 1}},
  { input: {r: 0.24, g:0.24, b:0.24}, output: { White: 1}},
  { input: {r: 255, g:255, b:255}, output: { White: 1}},
  { input: {r: 250, g:250, b:250}, output: { White: 1}},
  { input: {r: 245, g:245, b:245}, output: { White: 1}},
  { input: {r: 235, g:235, b:235}, output: { White: 1}},
  { input: {r: 0.211, g:0.232, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.239, g:0.247, b:0.255}, output: { Blue: 1}},
  { input: {r: 0.002, g:0.220, b:0.227}, output: { Blue: 1}},
  { input: {r: 0.016, g:0.049, b:0.183}, output: { Blue: 1}},
  { input: {r: 0.069, g:0.2, b:0.217}, output: { Blue: 1}},
  { input: {r: 0.148, g:0.223, b:0.257}, output: { Blue: 1}},
  { input: {r: 0.046, g:0.006, b:0.229}, output: { Blue: 1}},
  { input: {r: 0.13, g:0.02, b:0.029}, output: { Red: 1}},
  { input: {r: 0.16, g:0.051, b:0.086}, output: { Red: 1}},
  { input: {r: 0.19, g:0.18, b:0.14}, output: { Yellow: 1}},
  { input: {r: 0.231, g:0.232, b:0.077}, output: { Yellow: 1}},
])

let red = document.querySelector('.red')
let green = document.querySelector('.green')
let blue = document.querySelector('.blue')

let colorPicked = {r:red.value/1000, g:green.value/1000, b:blue.value/1000}
updateColor()

red.addEventListener('change', function() {
  colorPicked = {r:red.value/1000, g:green.value/1000, b:blue.value/1000}
  updateColor()
})

green.addEventListener('change', function() {
  colorPicked = {r:red.value/1000, g:green.value/1000, b:blue.value/1000}
  updateColor()
})

blue.addEventListener('change', function() {
  colorPicked = {r:red.value/1000, g:green.value/1000, b:blue.value/1000}
  updateColor()
})

function updateColor() {
  let output = net.run(colorPicked) // [0.987]
  for(let key in output) {
    console.log(key, output[key].toFixed(2))
  }
  console.log(colorPicked)
  let color = document.getElementById('color')
  let colorText = document.getElementById('color-text')
  color.style.backgroundColor = `rgb(${colorPicked.r*1000},${colorPicked.g*1000},${colorPicked.b*1000})`
  output = Object.keys(output).reduce((a, b) => output[a] > output[b] ? a : b)
  colorText.textContent = output
  colorText.style.color = (colorPicked.r*1000) + (colorPicked.g*1000) + (colorPicked.b*1000) > 400 ? `rgb(0 0 0)` : `rgb(255 255 255)`
}

//paint out all the colors on the canvas labeling the different colors areas
let canvas1 = document.getElementById('canvas1')
let ctx1 = canvas1.getContext('2d')
canvas1.width = 128
canvas1.height = 128

let canvas2 = document.getElementById('canvas2')
let ctx2 = canvas2.getContext('2d')
canvas2.width = 128
canvas2.height = 128

let canvas3 = document.getElementById('canvas3')
let ctx3 = canvas3.getContext('2d')
canvas3.width = 128
canvas3.height = 128

let canvas4 = document.getElementById('canvas4')
let ctx4 = canvas4.getContext('2d')
canvas4.width = 128
canvas4.height = 128

let canvas5 = document.getElementById('canvas5')
let ctx5 = canvas5.getContext('2d')
canvas5.width = 128
canvas5.height = 128

let canvas6 = document.getElementById('canvas6')
let ctx6 = canvas6.getContext('2d')
canvas6.width = 128
canvas6.height = 128

let canvas7 = document.getElementById('canvas7')
let ctx7 = canvas7.getContext('2d')
canvas7.width = 128
canvas7.height = 128

let canvas8 = document.getElementById('canvas8')
let ctx8 = canvas8.getContext('2d')
canvas8.width = 128
canvas8.height = 128

let canvas9 = document.getElementById('canvas9')
let ctx9 = canvas9.getContext('2d')
canvas9.width = 128
canvas9.height = 128

let canvas10 = document.getElementById('canvas10')
let ctx10 = canvas10.getContext('2d')
canvas10.width = 128
canvas10.height = 128

let canvas11 = document.getElementById('canvas11')
let ctx11 = canvas11.getContext('2d')
canvas11.width = 128
canvas11.height = 128

let canvas12 = document.getElementById('canvas12')
let ctx12 = canvas12.getContext('2d')
canvas12.width = 128
canvas12.height = 128



//paint the pixels in the canvas in all different colors and label them with the help of the neural network

let k = 0
for(let i = 256; i > 0; i-=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = net.run({r:j/1000, g:i/1000, b:0})
    //Kollar vilken färg som nätverket säger är mest sannolik och väljer den
    color = Object.keys(color).reduce((a, b) => color[a] > color[b] ? a : b)
    ctx1.fillStyle = color
    ctx1.fillRect(j/2, k/2, 1, 1)
  }
  k+=2
}

k = 0
for(let i = 256; i > 0; i-=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = {r:j, g:i, b:0}
    ctx2.fillStyle = `rgb(${color.r} ${color.g} ${color.b})`
    ctx2.fillRect(j/2, k/2, 1, 1)
  }
  k+=2
}

for(let i = 0; i < 256; i+=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = net.run({r:i/1000, g:0, b:j/1000})
        //Kollar vilken färg som nätverket säger är mest sannolik och väljer den
    color = Object.keys(color).reduce((a, b) => color[a] > color[b] ? a : b)
    ctx3.fillStyle = color
    ctx3.fillRect(i/2, j/2, 1, 1)
  }
}

for(let i = 0; i < 256; i+=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = {r:i, g:0, b:j}
    ctx4.fillStyle = `rgb(${color.r} ${color.g} ${color.b})`
    ctx4.fillRect(i/2, j/2, 1, 1)
  }
}

k = 256
for(let i = 0; i < 256; i+=2) {
  k-=2
  for(let j = 0; j < 256; j+=2) {
    let color = net.run({r:0, g:i/1000, b:j/1000})
        //Kollar vilken färg som nätverket säger är mest sannolik och väljer den
    color = Object.keys(color).reduce((a, b) => color[a] > color[b] ? a : b)
    ctx5.fillStyle = color
    ctx5.fillRect(k/2, j/2, 1, 1)
  }
}

k = 256
for(let i = 0; i < 256; i+=2) {
  k-=2
  for(let j = 0; j < 256; j+=2) {
    let color = {r:0, g:i, b:j}
    ctx6.fillStyle = `rgb(${color.r} ${color.g} ${color.b})`
    ctx6.fillRect(k/2, j/2, 1, 1)
  }
}

k = 256
for(let i = 0; i < 256; i+=2) {
  k-=2
  for(let j = 0; j < 256; j+=2) {
    let color = net.run({r:j/1000, g:i/1000, b:256/1000})
    //Kollar vilken färg som nätverket säger är mest sannolik och väljer den
    color = Object.keys(color).reduce((a, b) => color[a] > color[b] ? a : b)
    ctx7.fillStyle = color
    ctx7.fillRect(j/2,i/2, 1, 1)
  }
}

k = 256
for(let i = 0; i < 256; i+=2) {
  k-=2
  for(let j = 0; j < 256; j+=2) {
    let color = {r:j, g:i, b:256}
    ctx8.fillStyle = `rgb(${color.r} ${color.g} ${color.b})`
    ctx8.fillRect(j/2,i/2, 1, 1)
  }
}

k=0
for(let i = 256; i > 0; i-=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = net.run({r:j/1000, g:256/1000, b:i/1000})
    //Kollar vilken färg som nätverket säger är mest sannolik och väljer den
    color = Object.keys(color).reduce((a, b) => color[a] > color[b] ? a : b)
    ctx9.fillStyle = color
    ctx9.fillRect(j/2, k/2, 1, 1)
  }
  k+=2
}

k=0
for(let i = 256; i > 0; i-=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = {r:j, g:256, b:i}
    ctx10.fillStyle = `rgb(${color.r} ${color.g} ${color.b})`
    ctx10.fillRect(j/2, k/2, 1, 1)
  }
  k+=2
}

k = 256
for(let i = 0; i < 256; i+=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = net.run({r:256/1000, g:i/1000, b:j/1000})
    //Kollar vilken färg som nätverket säger är mest sannolik och väljer den
    color = Object.keys(color).reduce((a, b) => color[a] > color[b] ? a : b)
    ctx11.fillStyle = color
    ctx11.fillRect(i/2, j/2, 1, 1)
  }
  k-=2
}

k = 256
for(let i = 0; i < 256; i+=2) {
  for(let j = 0; j < 256; j+=2) {
    let color = {r:256, g:i, b:j}
    ctx12.fillStyle = `rgb(${color.r} ${color.g} ${color.b})`
    ctx12.fillRect(i/2, j/2, 1, 1)
  }
  k-=2
}

document.getElementById('random').addEventListener('click', function() {
  colorPicked = {r:Math.random()/3.8, g:Math.random()/3.8, b:Math.random()/3.8}
  red.value = colorPicked.r*1000
  green.value = colorPicked.g*1000
  blue.value = colorPicked.b*1000
  updateColor()
})

