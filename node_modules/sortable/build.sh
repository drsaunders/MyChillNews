#! /usr/bin/env bash
{
cat src/prefix.js
echo ';'
cat src/jquery-ui-1.10.1.custom.js
echo ';'
cat src/jquery.ui.touch-punch.js
} > ui.js

browserify test.js --debug > static/bundle.js
