/* Golden SIXS calibration vectors from binoculars-ng 7be4f79:
 * hkl_binoculars_detector_2d_sixs_calibration on the coordinates of every
 * detector, normalize_flag = 0. Each case dumps the positions of the four
 * corner pixels and the central pixel before ("before") and after
 * ("pixels") the calibration.
 * Build and run: bin_7be4f79_sixs.sh */
#include <stdio.h>
#include <stdlib.h>
#include "hkl-binoculars.h"

static const int dets[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const struct { double fx, fy; double sdd, detrot; } cases[] = {
	{0.5, 0.5, 1.0, 0.0},
	{0.3, 0.7, 0.9, 0.35},
	{0.0, 0.0, 1.2, -0.8},
	{0.9, 0.2, 0.5, 1.5707963267948966},
};

int main(void)
{
	int first = 1;
	printf("{\"cases\":[");
	for (size_t d = 0; d < sizeof dets / sizeof *dets; ++d)
		for (size_t c = 0; c < sizeof cases / sizeof *cases; ++c) {
			int w, h;
			hkl_binoculars_detector_2d_shape_get(dets[d], &w, &h);
			size_t n = (size_t)w * h;
			int ix0 = (int)(cases[c].fx * (w - 1)), iy0 = (int)(cases[c].fy * (h - 1));
			double *a = hkl_binoculars_detector_2d_coordinates_get(dets[d]);
			int px[5][2] = {{0, 0}, {w - 1, 0}, {0, h - 1}, {w - 1, h - 1}, {ix0, iy0}};
			printf("%s{\"id\":%d,\"ix0\":%d,\"iy0\":%d,\"sdd\":%.17g,\"detrot\":%.17g,",
			       first ? "" : ",", dets[d], ix0, iy0, cases[c].sdd, cases[c].detrot);
			first = 0;
			for (int pass = 0; pass < 2; ++pass) {
				if (pass)
					hkl_binoculars_detector_2d_sixs_calibration(dets[d], a, w, h, ix0, iy0,
										    cases[c].sdd, cases[c].detrot, 0);
				printf("%s\"%s\":[", pass ? "," : "", pass ? "pixels" : "before");
				for (int k = 0; k < 5; ++k) {
					size_t i = (size_t)px[k][1] * w + px[k][0];
					printf("%s[%d,%d,%.17g,%.17g,%.17g]", k ? "," : "", px[k][1], px[k][0],
					       a[i], a[n + i], a[2 * n + i]);
				}
				printf("]");
			}
			printf("}");
			free(a);
		}
	printf("]}\n");
	return 0;
}
