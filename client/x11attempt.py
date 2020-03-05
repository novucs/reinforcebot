from Xlib.display import Display

display = Display()
window = display.screen().root
result = window.query_pointer()
g = result.child.get_geometry()
x, y, w, h = g.x, g.y, g.width, g.height
print(result.child.id, x, y, w, h)

# this may help:
# https://github.com/MaartenBaert/ssr/blob/1c1edd6262a788b624b44efa5648b79deab8ae25/src/GUI/PageInput.cpp#L735
