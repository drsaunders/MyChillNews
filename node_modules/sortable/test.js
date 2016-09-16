
var h = require('hyperscript')

function template (s) {
   var e = document.createElement('h1')
   e.innerText = s
   return e
}

var array = 'Apple,Banana,Cherry,Durian,ElderBerry'.split(',')
var cur, log

var emitter = require('./')(array, template).on('change', ch)

function ch(v, ch) {
  console.log('CH', v, ch)
  cur.innerText = JSON.stringify(v, null, 2)
  log.innerText += '\n' + JSON.stringify(ch)
}

function button(name, fun) {
  if(!fun) fun = name, name = null
  return h('button', {onclick: fun}, name || fun.name || 'button')
}

var input

document.body.appendChild(
  h('div#content', 
    h('div',
      input = h('input', {onchange: function () {
        if(this.value) {
  //        emitter.push(this.value)
  //        this.value = ''
          this.select()
        }
      }}),
      h('span', {style: {display: 'inline-block'}},
        button(function shift () {
          input.value = emitter.shift()
        }),
        button(function unshift () {
          if(input.value)
            emitter.unshift(input.value)
          input.value = ''
        }),
        button(function pop () {
          input.value = emitter.pop()
        }),
        button(function push () {
          if(input.value)
            emitter.push(input.value)
          input.value = ''
        })
      )
    ),
    emitter.element,
    h('div',
      h('h2', 'current:'),
      cur = h('pre'),
      h('h2', 'history:'),
      log = h('pre')
    )
  )
)

ch(array, [])

