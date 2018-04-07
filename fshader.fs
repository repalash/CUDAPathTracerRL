#version 120

uniform sampler2D texImage;
varying vec2 fTexCoord;

void main(void) {
	gl_FragColor = texture2D(texImage, fTexCoord);
}
