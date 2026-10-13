/**
 * photoAreaGroundTruthData.js — Ground-truth pixel bordering templates
 * derived directly from photo_area.jpg human annotations.
 */

export const PHOTO_AREA_METADATA = {
  source: 'photo_area.jpg',
  format: 'dental-ar-gt/1',
  description: 'Human-verified sub-pixel tooth boundary polygons from photo_area.jpg dual-card ground truth',
  upperTeethCount: 14,
  lowerTeethCount: 14,
};

export const UPPER_ARCH_GT = [
  {
    "fdi": 11,
    "tooth_type": "central_incisor",
    "group": "incisor",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": -0.0485,
      "v": -0.0552
    },
    "contour": [
      [
        -0.097,
        -0.1079
      ],
      [
        0.0005,
        -0.109
      ],
      [
        -0.0019,
        -0.0026
      ],
      [
        -0.0959,
        -0.0016
      ]
    ]
  },
  {
    "fdi": 21,
    "tooth_type": "central_incisor",
    "group": "incisor",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": 0.0485,
      "v": -0.0531
    },
    "contour": [
      [
        0.0005,
        -0.109
      ],
      [
        0.0981,
        -0.1011
      ],
      [
        0.0939,
        0.0
      ],
      [
        0.0016,
        -0.0026
      ]
    ]
  },
  {
    "fdi": 12,
    "tooth_type": "lateral_incisor",
    "group": "incisor",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": -0.1391,
      "v": -0.0291
    },
    "contour": [
      [
        -0.1851,
        -0.0627
      ],
      [
        -0.0969,
        -0.0991
      ],
      [
        -0.0958,
        0.0073
      ],
      [
        -0.1788,
        0.0383
      ]
    ]
  },
  {
    "fdi": 22,
    "tooth_type": "lateral_incisor",
    "group": "incisor",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": 0.1309,
      "v": -0.0226
    },
    "contour": [
      [
        0.0982,
        -0.0958
      ],
      [
        0.1696,
        -0.0487
      ],
      [
        0.1617,
        0.049
      ],
      [
        0.0939,
        0.0053
      ]
    ]
  },
  {
    "fdi": 13,
    "tooth_type": "canine",
    "group": "canine",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": -0.2152,
      "v": 0.0285
    },
    "contour": [
      [
        -0.2554,
        0.0001
      ],
      [
        -0.1851,
        -0.0627
      ],
      [
        -0.175,
        0.0613
      ],
      [
        -0.2453,
        0.1152
      ]
    ]
  },
  {
    "fdi": 23,
    "tooth_type": "canine",
    "group": "canine",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": 0.1848,
      "v": 0.0474
    },
    "contour": [
      [
        0.1696,
        -0.0487
      ],
      [
        0.2184,
        0.0395
      ],
      [
        0.1982,
        0.1461
      ],
      [
        0.1529,
        0.0526
      ]
    ]
  },
  {
    "fdi": 14,
    "tooth_type": "premolar_1",
    "group": "premolar",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": -0.2616,
      "v": 0.1265
    },
    "contour": [
      [
        -0.2987,
        0.098
      ],
      [
        -0.232,
        0.0353
      ],
      [
        -0.2272,
        0.1594
      ],
      [
        -0.2887,
        0.2132
      ]
    ]
  },
  {
    "fdi": 24,
    "tooth_type": "premolar_1",
    "group": "premolar",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": 0.2072,
      "v": 0.1571
    },
    "contour": [
      [
        0.1975,
        0.0752
      ],
      [
        0.2337,
        0.1546
      ],
      [
        0.2205,
        0.2434
      ],
      [
        0.177,
        0.1552
      ]
    ]
  },
  {
    "fdi": 15,
    "tooth_type": "premolar_2",
    "group": "premolar",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": -0.2983,
      "v": 0.2244
    },
    "contour": [
      [
        -0.3332,
        0.1959
      ],
      [
        -0.2718,
        0.1332
      ],
      [
        -0.2652,
        0.2573
      ],
      [
        -0.3231,
        0.3111
      ]
    ]
  },
  {
    "fdi": 25,
    "tooth_type": "premolar_2",
    "group": "premolar",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": 0.2029,
      "v": 0.2546
    },
    "contour": [
      [
        0.1896,
        0.1727
      ],
      [
        0.2312,
        0.2521
      ],
      [
        0.218,
        0.3409
      ],
      [
        0.1727,
        0.2527
      ]
    ]
  },
  {
    "fdi": 16,
    "tooth_type": "molar_1",
    "group": "molar",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": -0.3229,
      "v": 0.3266
    },
    "contour": [
      [
        -0.3588,
        0.2937
      ],
      [
        -0.2938,
        0.231
      ],
      [
        -0.2871,
        0.3639
      ],
      [
        -0.3521,
        0.4177
      ]
    ]
  },
  {
    "fdi": 26,
    "tooth_type": "molar_1",
    "group": "molar",
    "jaw": "upper",
    "partial": false,
    "center": {
      "u": 0.1903,
      "v": 0.3611
    },
    "contour": [
      [
        0.1729,
        0.2704
      ],
      [
        0.2216,
        0.3497
      ],
      [
        0.2103,
        0.4562
      ],
      [
        0.1562,
        0.3681
      ]
    ]
  },
  {
    "fdi": 17,
    "tooth_type": "molar_2",
    "group": "molar",
    "jaw": "upper",
    "partial": true,
    "center": {
      "u": -0.3462,
      "v": 0.4376
    },
    "contour": [
      [
        -0.3807,
        0.4003
      ],
      [
        -0.3174,
        0.3465
      ],
      [
        -0.3125,
        0.4794
      ],
      [
        -0.3741,
        0.5243
      ]
    ]
  },
  {
    "fdi": 27,
    "tooth_type": "molar_2",
    "group": "molar",
    "jaw": "upper",
    "partial": true,
    "center": {
      "u": 0.1727,
      "v": 0.4655
    },
    "contour": [
      [
        0.1563,
        0.377
      ],
      [
        0.2014,
        0.4563
      ],
      [
        0.1936,
        0.5539
      ],
      [
        0.1396,
        0.4747
      ]
    ]
  }
];

export const LOWER_ARCH_GT = [
  {
    "fdi": 41,
    "tooth_type": "central_incisor",
    "group": "incisor",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": -0.0396,
      "v": 0.0493
    },
    "contour": [
      [
        -0.0791,
        0.0024
      ],
      [
        0.0001,
        0.0059
      ],
      [
        -0.0017,
        0.0962
      ],
      [
        -0.0777,
        0.0926
      ]
    ]
  },
  {
    "fdi": 31,
    "tooth_type": "central_incisor",
    "group": "incisor",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": 0.0396,
      "v": 0.0481
    },
    "contour": [
      [
        0.0001,
        0.0059
      ],
      [
        0.0792,
        0.0
      ],
      [
        0.0806,
        0.0902
      ],
      [
        -0.0017,
        0.0962
      ]
    ]
  },
  {
    "fdi": 42,
    "tooth_type": "lateral_incisor",
    "group": "incisor",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": -0.1159,
      "v": 0.0314
    },
    "contour": [
      [
        -0.1556,
        -0.0282
      ],
      [
        -0.0791,
        0.0024
      ],
      [
        -0.0777,
        0.0926
      ],
      [
        -0.1511,
        0.0589
      ]
    ]
  },
  {
    "fdi": 32,
    "tooth_type": "lateral_incisor",
    "group": "incisor",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": 0.1181,
      "v": 0.0279
    },
    "contour": [
      [
        0.0792,
        0.0
      ],
      [
        0.1579,
        -0.0329
      ],
      [
        0.1545,
        0.0543
      ],
      [
        0.0806,
        0.0902
      ]
    ]
  },
  {
    "fdi": 43,
    "tooth_type": "canine",
    "group": "canine",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": -0.189,
      "v": -0.0118
    },
    "contour": [
      [
        -0.2276,
        -0.0825
      ],
      [
        -0.1524,
        -0.0282
      ],
      [
        -0.1543,
        0.0589
      ],
      [
        -0.2216,
        0.0045
      ]
    ]
  },
  {
    "fdi": 33,
    "tooth_type": "canine",
    "group": "canine",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": 0.191,
      "v": -0.0175
    },
    "contour": [
      [
        0.1579,
        -0.0329
      ],
      [
        0.2268,
        -0.0893
      ],
      [
        0.2249,
        -0.0022
      ],
      [
        0.1545,
        0.0543
      ]
    ]
  },
  {
    "fdi": 44,
    "tooth_type": "premolar_1",
    "group": "premolar",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": -0.2573,
      "v": -0.0821
    },
    "contour": [
      [
        -0.2922,
        -0.1607
      ],
      [
        -0.2199,
        -0.0906
      ],
      [
        -0.2264,
        -0.0033
      ],
      [
        -0.2909,
        -0.0736
      ]
    ]
  },
  {
    "fdi": 34,
    "tooth_type": "premolar_1",
    "group": "premolar",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": 0.2545,
      "v": -0.0897
    },
    "contour": [
      [
        0.2203,
        -0.0971
      ],
      [
        0.2889,
        -0.1694
      ],
      [
        0.287,
        -0.0823
      ],
      [
        0.2216,
        -0.0101
      ]
    ]
  },
  {
    "fdi": 45,
    "tooth_type": "premolar_2",
    "group": "premolar",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": -0.3071,
      "v": -0.1803
    },
    "contour": [
      [
        -0.3365,
        -0.263
      ],
      [
        -0.2719,
        -0.1848
      ],
      [
        -0.2801,
        -0.0976
      ],
      [
        -0.3399,
        -0.1758
      ]
    ]
  },
  {
    "fdi": 35,
    "tooth_type": "premolar_2",
    "group": "premolar",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": 0.3037,
      "v": -0.1895
    },
    "contour": [
      [
        0.2727,
        -0.193
      ],
      [
        0.3349,
        -0.2731
      ],
      [
        0.333,
        -0.1859
      ],
      [
        0.274,
        -0.1059
      ]
    ]
  },
  {
    "fdi": 46,
    "tooth_type": "molar_1",
    "group": "molar",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": -0.3317,
      "v": -0.2868
    },
    "contour": [
      [
        -0.3587,
        -0.3735
      ],
      [
        -0.2989,
        -0.2952
      ],
      [
        -0.3038,
        -0.2001
      ],
      [
        -0.3652,
        -0.2784
      ]
    ]
  },
  {
    "fdi": 36,
    "tooth_type": "molar_1",
    "group": "molar",
    "jaw": "lower",
    "partial": false,
    "center": {
      "u": 0.3258,
      "v": -0.2967
    },
    "contour": [
      [
        0.2964,
        -0.3042
      ],
      [
        0.3538,
        -0.3842
      ],
      [
        0.3552,
        -0.2892
      ],
      [
        0.2978,
        -0.2092
      ]
    ]
  },
  {
    "fdi": 47,
    "tooth_type": "molar_2",
    "group": "molar",
    "jaw": "lower",
    "partial": true,
    "center": {
      "u": -0.3461,
      "v": -0.4014
    },
    "contour": [
      [
        -0.3716,
        -0.4921
      ],
      [
        -0.3164,
        -0.4058
      ],
      [
        -0.3213,
        -0.3107
      ],
      [
        -0.3749,
        -0.397
      ]
    ]
  },
  {
    "fdi": 37,
    "tooth_type": "molar_2",
    "group": "molar",
    "jaw": "lower",
    "partial": true,
    "center": {
      "u": 0.3356,
      "v": -0.4116
    },
    "contour": [
      [
        0.3106,
        -0.4152
      ],
      [
        0.3631,
        -0.5031
      ],
      [
        0.3613,
        -0.4081
      ],
      [
        0.3072,
        -0.3202
      ]
    ]
  }
];
