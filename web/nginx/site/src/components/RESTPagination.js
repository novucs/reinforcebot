import React, {Component} from "react";
import {Pagination} from "semantic-ui-react";


export default class RESTPagination extends Component {
  constructor(props) {
    // props:
    // itemCount: int
    // pageSize: int
    // onPageChange(totalPages, currentPage)
    // hideIfOnePage
    super(props);
    this.state = {currentPage: 1}
  }

  totalPages = () => {
    return Math.max(1, Math.ceil(this.props.itemCount / this.props.pageSize));
  };

  setPage = (event, {activePage}) => {
    this.setState({currentPage: Math.ceil(activePage)}, () => {
      this.props.onPageChange(this.state.currentPage);
    });
  };

  render = () => {
    if (this.props.hideIfOnePage && this.totalPages() === 1) {
      return null;
    }

    return (
      <Pagination
        activePage={this.state.currentPage}
        totalPages={this.totalPages()}
        onPageChange={this.setPage}
      />
    );
  };
}
