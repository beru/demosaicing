
#include "demosaic.h"

// http://jp.mathworks.com/help/images/ref/demosaic.html

// ------
// grgrgr
// bgbgbg
// grgrgr
// bgbgbg
// ------
// +r+r+r
// ++++++
// +r+r+r
// ++++++
// ------
// g+g+g+
// +g+g+g
// g+g+g+
// +g+g+g
// ------
// ++++++
// b+b+b+
// ++++++
// b+b+b+

void demosaic_grbg(
	const uint16_t* pSrc,
	size_t width,
	size_t height,
	uint32_t* pDst
	)
{
	size_t offset0 = 0;
	size_t offset1 = width;
	size_t offset2 = width * 2;
	const uint16_t* pUp = pSrc;
	const uint16_t* pMi = pSrc + width;
	const uint16_t* pLo = pMi + width;
	uint32_t* pColor = pDst + width;
	uint16_t r, g, b;
	uint32_t r0, g0, b0;
	uint16_t
		ul, uc, ur, ur2,
		ml, mc, mr, mr2,
		ll, lc, lr, lr2
	;
	const size_t nShifts = 8;

	// g r g r g r
	// b g b g b g
	// g r g r g r
	// b g b g b g
	for (size_t y=1; y<height-1; y+=2) {
		ul = pUp[0]; uc = pUp[1]; ur = pUp[2];
		ml = pMi[0]; mc = pMi[1]; mr = pMi[2];
		ll = pLo[0]; lc = pLo[1]; lr = pLo[2];
		for (size_t x=1; x<width-1; x+=2) {
			r0 = uc + lc + 1;
			r = r0 >> (1 + nShifts);
			g = mc >> nShifts;
			b = (ml + mr + 1) >> (1 + nShifts);
			pColor[x] = r | (g << 8) | (b << 16); 
			
			ur2 = pUp[x + 2];
			mr2 = pMi[x + 2];
			lr2 = pLo[x + 2];
			
			r = (r0 + ur2 + lr2 + 1) >> (2 + nShifts);
			g = (mc + mr2 + 1) >> (1 + nShifts);
			b = mr >> nShifts;
			pColor[x + 1] = r | (g << 8) | (b << 16); 
			
			ul = ur;
			ml = mr;
			ll = lr;
			uc = ur2;
			mc = mr2;
			lc = lr2;
			ur = pUp[x + 3];
			mr = pMi[x + 3];
			lr = pLo[x + 3];
		}
		pUp += width;
		pMi += width;
		pLo += width;
		pColor += width;
		ul = pUp[0]; uc = pUp[1]; ur = pUp[2];
		ml = pMi[0]; mc = pMi[1]; mr = pMi[2];
		ll = pLo[0]; lc = pLo[1]; lr = pLo[2];
		for (size_t x=1; x<width-1; x+=2) {
			r = mc >> nShifts;
			g = (uc + ml + mr + lc + 2) >> (2 + nShifts);
			b = (ul + ur + ll + lr + 2) >> (2 + nShifts);
			pColor[x] = r | (g << 8) | (b << 16); 
			
			ur2 = pUp[x + 2];
			mr2 = pMi[x + 2];
			lr2 = pLo[x + 2];
			
			r = (mc + mr2 + 1) >> (1 + nShifts);
			g = mr >> nShifts;
			b = (ur + lr + 1) >> (1 + nShifts);
			pColor[x + 1] = r | (g << 8) | (b << 16); 
			
			ul = ur;
			ml = mr;
			ll = lr;
			uc = ur2;
			mc = mr2;
			lc = lr2;
			ur = pUp[x + 3];
			mr = pMi[x + 3];
			lr = pLo[x + 3];
		}
		pUp += width;
		pMi += width;
		pLo += width;
		pColor += width;
	}
	
}




