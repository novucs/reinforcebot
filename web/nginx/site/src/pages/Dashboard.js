import React from 'react';
import TopMenu from "../TopMenu";
import {Search} from "semantic-ui-react";
import Footer from "../Footer";
import {ensureSignedIn} from "../Util";
import _ from 'lodash'

const initialState = {isLoading: false, results: [], value: ''};
const source = ["blue", "red", "green"];

export default class Dashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = initialState;
  }

  handleResultSelect = (e, {result}) => this.setState({value: result.title});

  handleSearchChange = (e, {value}) => {
    this.setState({isLoading: true, value});

    setTimeout(() => {
      if (this.state.value.length < 1) return this.setState(initialState);

      const re = new RegExp(_.escapeRegExp(this.state.value), 'i');
      const isMatch = (result) => re.test(result.title);

      this.setState({
        isLoading: false,
        results: _.filter(source, isMatch),
      })
    }, 300);
  };

  render() {
    ensureSignedIn();

    return (
      <div className="SitePage">
        <TopMenu/>
        <div className="SiteContents" style={{backgroundColor: '#F7F7F7', marginTop: '64px'}}>
          <Search
            size="massive"
            loading={this.state.isLoading}
            onResultSelect={this.handleResultSelect}
            onSearchChange={_.debounce(this.handleSearchChange, 500, {
              leading: true,
            })}
            results={this.state.results}
            value={this.state.value}
          />
        </div>
        <Footer/>
      </div>
    );
  }
}
