// React Compiler runs as a post-pass over tsc's `dist/` output.
//
// tsc emits JSX as _jsx() calls (jsx: "react-jsx"). babel-plugin-react-compiler
// accepts that shape and auto-memoizes every component it recognizes via the
// default `infer` compilation mode (PascalCase components + use-prefixed
// hooks). The `sources` filter keeps it from walking node_modules files that
// end up in source maps.
//
// target=19 matches our react ^19.2.4 dependency.
module.exports = {
  assumptions: {
    setPublicClassFields: true
  },
  plugins: [
    [
      'babel-plugin-react-compiler',
      {
        target: '19',
        sources: (filename) => {
          if (!filename) return false
          if (filename.includes('node_modules')) return false
          return true
        }
      }
    ]
  ],
  // We feed already-compiled JS into babel; don't re-parse as TS/JSX.
  // @babel/preset-env etc. would over-transform — the compiler is our only
  // transform here. babelrc:false stops @babel/cli from walking up the
  // filesystem looking for other configs (the parent repo might add one).
  babelrc: false
}
