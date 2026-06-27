module.exports = {
  assumptions: {
    setPublicClassFields: true
  },
  plugins: [
    [
      'babel-plugin-react-compiler',
      {
        target: '19',
        sources: filename => Boolean(filename && !filename.includes('node_modules'))
      }
    ]
  ],
  babelrc: false
}
