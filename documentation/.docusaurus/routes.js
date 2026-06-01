import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/',
    component: ComponentCreator('/', 'fbe'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', '93c'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', '9a8'),
            routes: [
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
