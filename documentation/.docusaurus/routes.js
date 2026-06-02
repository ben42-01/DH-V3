import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/DH-V3/__docusaurus/debug',
    component: ComponentCreator('/DH-V3/__docusaurus/debug', '0e4'),
    exact: true
  },
  {
    path: '/DH-V3/__docusaurus/debug/config',
    component: ComponentCreator('/DH-V3/__docusaurus/debug/config', '3b2'),
    exact: true
  },
  {
    path: '/DH-V3/__docusaurus/debug/content',
    component: ComponentCreator('/DH-V3/__docusaurus/debug/content', '7b3'),
    exact: true
  },
  {
    path: '/DH-V3/__docusaurus/debug/globalData',
    component: ComponentCreator('/DH-V3/__docusaurus/debug/globalData', 'b81'),
    exact: true
  },
  {
    path: '/DH-V3/__docusaurus/debug/metadata',
    component: ComponentCreator('/DH-V3/__docusaurus/debug/metadata', 'f72'),
    exact: true
  },
  {
    path: '/DH-V3/__docusaurus/debug/registry',
    component: ComponentCreator('/DH-V3/__docusaurus/debug/registry', 'eb4'),
    exact: true
  },
  {
    path: '/DH-V3/__docusaurus/debug/routes',
    component: ComponentCreator('/DH-V3/__docusaurus/debug/routes', '831'),
    exact: true
  },
  {
    path: '/DH-V3/',
    component: ComponentCreator('/DH-V3/', '843'),
    routes: [
      {
        path: '/DH-V3/',
        component: ComponentCreator('/DH-V3/', 'c1b'),
        routes: [
          {
            path: '/DH-V3/',
            component: ComponentCreator('/DH-V3/', '5d4'),
            routes: [
              {
                path: '/DH-V3/abstract',
                component: ComponentCreator('/DH-V3/abstract', '50b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/core-concepts',
                component: ComponentCreator('/DH-V3/core-concepts', '81e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments',
                component: ComponentCreator('/DH-V3/experiments', '7d7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-01',
                component: ComponentCreator('/DH-V3/experiments/exp-01', '1a7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-02',
                component: ComponentCreator('/DH-V3/experiments/exp-02', 'c25'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-03',
                component: ComponentCreator('/DH-V3/experiments/exp-03', '54b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-04',
                component: ComponentCreator('/DH-V3/experiments/exp-04', 'fa2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-05',
                component: ComponentCreator('/DH-V3/experiments/exp-05', 'e8b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-06',
                component: ComponentCreator('/DH-V3/experiments/exp-06', 'c65'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-07',
                component: ComponentCreator('/DH-V3/experiments/exp-07', '79f'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/experiments/exp-08',
                component: ComponentCreator('/DH-V3/experiments/exp-08', '911'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/intro',
                component: ComponentCreator('/DH-V3/intro', 'd79'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/metrics-reference',
                component: ComponentCreator('/DH-V3/metrics-reference', 'f1a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/results-interpretation',
                component: ComponentCreator('/DH-V3/results-interpretation', 'fb7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/running-experiments',
                component: ComponentCreator('/DH-V3/running-experiments', '2eb'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/DH-V3/',
                component: ComponentCreator('/DH-V3/', 'b0f'),
                exact: true,
                sidebar: "docsSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
