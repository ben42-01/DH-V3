import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', 'b36'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', 'e2c'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', '417'),
            routes: [
              {
                path: '/abstract',
                component: ComponentCreator('/abstract', '002'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/core-concepts',
                component: ComponentCreator('/core-concepts', 'ee7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/experiments',
                component: ComponentCreator('/experiments', '03c'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/experiments/exp-01',
                component: ComponentCreator('/experiments/exp-01', '50b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/experiments/exp-02',
                component: ComponentCreator('/experiments/exp-02', 'eec'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/experiments/exp-03',
                component: ComponentCreator('/experiments/exp-03', '8a8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/experiments/exp-04',
                component: ComponentCreator('/experiments/exp-04', '105'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/intro',
                component: ComponentCreator('/intro', '4a2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/metrics-reference',
                component: ComponentCreator('/metrics-reference', 'eed'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/results-interpretation',
                component: ComponentCreator('/results-interpretation', 'a2a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/running-experiments',
                component: ComponentCreator('/running-experiments', '997'),
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
