import React from 'react';
import {ensureSignedOut, fetchMe, signIn} from '../Util';
import {Button, Form, Grid, Header, Message, Segment} from "semantic-ui-react";
import logo from '../icon.svg'
import TopMenu from "../components/TopMenu";
import Footer from "../components/Footer";

export default class SignIn extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      username: '',
      password: '',
      errors: [],
    }
  }

  submit = (event) => {
    signIn(this.state.username, this.state.password, (errors) => {
      this.setState({errors: errors});
    });
  };

  keyPress = (event) => {
    if (event.keyCode === 13 && this.ableToSubmit(event)) {
      this.submit(event);
    }
  };

  ableToSubmit = (event) => {
    return this.state.username !== ''
      && this.state.password !== '';
  };

  componentDidMount = () => {
    ensureSignedOut();
    fetchMe(me => this.setState({me}));
  };

  render = () => (
    <div className='SitePage'>
      <TopMenu me={this.state.me}/>
      <Grid textAlign='center' style={{marginTop: '32px', marginBottom: '32px'}} className='SiteContents'
            verticalAlign='middle'>
        <Grid.Column style={{maxWidth: 450}}>
          <Header as="h2" color="teal" textAlign="center">
            <img src={logo} alt="logo" className="image"/>{" "}
            Sign-in to your account
          </Header>
          <Form size="large">
            <Segment stacked>
              <Form.Input
                fluid
                icon="user"
                iconPosition="left"
                placeholder="Username"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({username: event.target.value})}
              />
              <Form.Input
                fluid
                icon="lock"
                iconPosition="left"
                placeholder="Password"
                type="password"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({password: event.target.value})}
              />
              <Button
                color="teal"
                fluid size="large"
                disabled={!this.ableToSubmit()}
                onClick={this.submit}
              >
                Sign in
              </Button>
            </Segment>
          </Form>
          <Message
            error
            header='Sign In Unsuccessful'
            list={this.state.errors}
            hidden={this.state.errors.length === 0}
          />
          <Message info>
            New to us? <a href="/signup">Sign Up</a>
          </Message>
        </Grid.Column>
      </Grid>
      <Footer/>
    </div>
  );
}
