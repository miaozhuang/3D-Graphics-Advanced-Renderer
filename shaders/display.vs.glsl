attribute vec2 pos;// quad vetices
varying vec2 uv; 
void main(void) {
	gl_Position =  vec4(pos, 0, 1.0);
	uv = (pos + 1.)/2.;
}