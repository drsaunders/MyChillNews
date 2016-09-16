# drag-sortable

Create a list that is can be sorted by dragging ui elements.

``` js
var Sortable = require('./')

var array = ['Apple', 'Banana', 'Cherry', 'Durian', 'ElderBerry']
var emitter = Sortable(array, function template (s) {
  //return html Element.
  var e = document.createElement('h1')
  e.innerText = s
  return e
}, document.createElement('ol'))
//listen to changes (newArray, splices)
emitter.on('change', console.log.bind(console))

emitter.element
```

Change events is the new state of the array,
and the splices that changed the array.

`template` is just a function that takes any thing and returns an `HTMLElement`


## jquery-ui

Currently, this depends on jquery UI, which is not ideal,
but it works, and it works on mobile.

## known issues:

Dragging goes funny when zoomed on Chrome Android.
Be sure to use viewport:

```
<meta name="viewport"
  content="width=device-width, initial-scale=1, minimum-scale=1, maximum-scale=1">
```

To disable zoom.
