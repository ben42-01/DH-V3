/** @type {import('@docusaurus/sidebar-utils').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    'abstract',
    'intro',
    'core-concepts',
    'running-experiments',
    'metrics-reference',
    'results-interpretation',
    {
      type: 'category',
      label: 'Research Experiments',
      items: [
        'experiments',
        {
          type: 'category',
          label: 'Experiment Reports',
          items: [
            'experiments/exp-01',
            'experiments/exp-02',
            'experiments/exp-03',
            'experiments/exp-04',
          ],
        },
      ],
    }
  ],
};

module.exports = sidebars;
