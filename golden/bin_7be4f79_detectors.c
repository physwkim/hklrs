/* Golden detector table from binoculars-ng 7be4f79: for every detector its
 * name, shape (width x height), pixel positions and mask, straight from
 * hkl_binoculars_detector_2d_*. Detectors whose positions are separable
 * (x = 0, y per column, z per row) are dumped as the two profiles; the
 * driver aborts if that claim is false for a pixel. The others (Cirpad) are
 * dumped at a fixed sample of pixels. Masks are run-length encoded over the
 * row-major array as [start, length]. Build and run: bin_7be4f79_detectors.sh */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hkl-binoculars.h"

static void sample(const double *a, int w, int h, int j, int i, int *first)
{
	size_t n = (size_t)w * h;
	printf("%s[%d,%d,%.17g,%.17g,%.17g]", *first ? "" : ",", j, i,
	       a[j * w + i], a[n + j * w + i], a[2 * n + j * w + i]);
	*first = 0;
}

int main(void)
{
	int nd = hkl_binoculars_detector_2d_number_of_detectors();
	printf("{\"detectors\":[");
	for (int d = 0; d < nd; ++d) {
		int w, h;
		hkl_binoculars_detector_2d_shape_get(d, &w, &h);
		size_t n = (size_t)w * h;
		const char *name = hkl_binoculars_detector_2d_name_get(d);
		double *c = hkl_binoculars_detector_2d_coordinates_get(d);
		uint8_t *m = hkl_binoculars_detector_2d_mask_get(d);
		int sep = 1;
		for (int j = 0; j < h && sep; ++j)
			for (int i = 0; i < w; ++i)
				if (c[j * w + i] != 0.0 || c[n + j * w + i] != c[n + i]
				    || c[2 * n + j * w + i] != c[2 * n + j * w])
					sep = 0;
		printf("%s{\"id\":%d,\"name\":\"%s\",\"width\":%d,\"height\":%d,", d ? "," : "", d, name, w, h);
		if (sep) {
			printf("\"y\":[");
			for (int i = 0; i < w; ++i)
				printf("%s%.17g", i ? "," : "", c[n + i]);
			printf("],\"z\":[");
			for (int j = 0; j < h; ++j)
				printf("%s%.17g", j ? "," : "", c[2 * n + j * w]);
			printf("],");
		} else {
			/* Cirpad: 20 modules of 120 rows x 560 columns (7 chips of 80) */
			static const int rows[] = {0, 1, 2, 3, 58, 59, 60, 61, 116, 117, 118, 119};
			static const int cols[] = {0, 1, 2, 40, 77, 78, 79};
			int first = 1;
			if (h != 2400 || w != 560) { fprintf(stderr, "unexpected non-separable %s\n", name); return 1; }
			printf("\"samples\":[");
			for (int k = 0; k < 20; ++k)
				for (size_t r = 0; r < sizeof rows / sizeof *rows; ++r)
					for (int ch = 0; ch < 7; ++ch)
						for (size_t q = 0; q < sizeof cols / sizeof *cols; ++q)
							sample(c, w, h, k * 120 + rows[r], ch * 80 + cols[q], &first);
			printf("],");
		}
		printf("\"mask_runs\":[");
		int first = 1;
		for (size_t k = 0; k < n;) {
			if (!m[k]) { ++k; continue; }
			size_t s = k;
			while (k < n && m[k]) ++k;
			printf("%s[%zu,%zu]", first ? "" : ",", s, k - s);
			first = 0;
		}
		printf("]}");
		free(c);
		free(m);
	}
	printf("]}\n");
	return 0;
}
