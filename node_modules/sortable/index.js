var $ = require('./ui')
var EventEmitter = require('events').EventEmitter

module.exports = function (array, template, list) {

  var emitter = new EventEmitter()
  if(!(list instanceof HTMLElement))
    list = document.createElement(list || "ol")

  function index (e) {
    return [].indexOf.call(e.parentNode.children, e)
  }

  sortableList(array, template, list) 

  function length (a) {
    return 'function' === typeof a ? a.length() : a.length
  }

  function sortableList(array, template, list) {
    array.forEach(function (e, i) {
      var el = e && template(e, i)
      if(el) list.appendChild(el)
    })

    var from
    
    $(list).sortable({
      start: function (e, u) {
        from = index(u.item[0])
        //console.log('START', from)
      },
      stop: function (e, u) {
        var to = index(u.item[0])
        //console.log('TO', to, 'FROM', from)
        //if(to > from) to
        var v = array[from]
        var changes = [[from, 1], [to, 0, v]]
        
        changes.forEach(function (ch) {
          array.splice.apply(array, ch)
          emitter.emit('change', array, ch)
          emitter.emit('splice', ch)
        })
      }
    }).disableSelection();
  }

  emitter.element = list
  emitter.splice = function (index, del) {
    var insert = [].slice.call(arguments, 2)
    var args = [].slice.call(arguments)
    _del = del || 0

    function at(i) {
      return list.children[i]
    }

    while(_del--)
      list.removeChild(at(index))

    insert.forEach(function (e, i) {
      var t = template(e, index + i), a = at(index)
      if(!t) return
      if(a) list.insertBefore(t, a)
      else  list.appendChild(t)
    })
    
    var r = array.splice.apply(array, args)
    emitter.emit('change', array, args)
    emitter.emit('splice', args)
    return r
  }

  emitter.unshift = function (o) {
    return emitter.splice(0, 0, o), length(array)
  }
  emitter.push = function (o) {
    return emitter.splice(length(array), 0, o), length(array)
  }
  emitter.shift = function (o) {
    return emitter.splice(0, 1)[0] || null
  }
  emitter.pop = function (o) {
    return emitter.splice(length(array) - 1, 1)[0]
  }
  emitter.length = function () {
    return length(array)
  }
  emitter.slice = function () {
    return array.slice.apply(array, arguments)
  }

  //this is just a hack, need better solution
  emitter._reset = function (ary) {
    list.innerHTML = '', array.length = 0
    ary.forEach(function (e, i) {
      var t = template(e, i)
      if(!t) return
      array[i] = e
      list.appendChild(t)
    })
    return emitter
  }

  return emitter
}
