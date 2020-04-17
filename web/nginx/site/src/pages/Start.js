import React from 'react';
import {Button, Container, Header} from "semantic-ui-react";
import logo from '../icon.svg'
import TopMenu from "../components/TopMenu";
import Footer from "../components/Footer";
import download from '../clientbinary'
import {fetchMe} from "../Util";

export default class Start extends React.Component {
  constructor(props) {
    super(props);
    this.state = {}
  }

  componentDidMount = () => {
    fetchMe(me => this.setState({me}));
  };

  render = () => (
    <div className='SitePage'>
      <TopMenu me={this.state.me}/>
      <Container text className='SiteContents' style={{marginTop: '7em', textAlign: 'left'}}>
        <Header as="h2" color="teal">
          <img src={logo} alt="logo" className="image"/>{" "}
          Getting Started
        </Header>
        <p>Currently only Xorg on Linux is supported.</p>
        <p>Download and run the below executable.</p>
        <Button
          primary
          download
          href={download}
          icon='download'
          content='Download'
          size='medium'
        />
      </Container>
      <Footer/>
    </div>
  );
}
